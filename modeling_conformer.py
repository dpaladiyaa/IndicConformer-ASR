from datetime import timedelta
import gc
import json
from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from torch import nn
from transformers import Wav2Vec2ConformerModel
from torch_state_bridge import state_bridge
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file
import webrtcvad
from torch.utils.data import Dataset , DataLoader
import srt

def calc_length(lengths, all_paddings=2, kernel_size=3, stride=2, repeat_num=1):
    add_pad = all_paddings - kernel_size
    for _ in range(repeat_num):
        lengths = torch.floor((lengths.float() + add_pad) / stride + 1)
    return lengths

class ChunkedData(Dataset):
    def __init__(self, wav, sr):
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0, keepdim=True)
        self.data, self.ts = self.make_chunks(wav)

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.ts[i]

    def make_chunks(self, wav, sr=16000, ag=2, min_s=10, max_s=15, ms=30):
        w = (wav * 32768).clamp(-32768, 32767).short().squeeze(0)
        fl = int(sr * ms / 1000)
        nf = len(w) // fl
        w = w[: nf * fl]
        fr = w.view(nf, fl)
        vad = webrtcvad.Vad(ag)
        sp = torch.zeros(nf, dtype=torch.bool)
        for i, f in enumerate(fr):
            try: sp[i] = vad.is_speech(f.cpu().numpy().tobytes(), sr)
            except: pass
        seg, s = [], None
        for i, v in enumerate(sp):
            if v and s is None: s = i
            elif not v and s is not None: seg.append((s, i)); s = None
        if s is not None: seg.append((s, len(sp)))
        cs, ts, st = [], [], 0
        mn, mx, N = int(min_s * sr), int(max_s * sr), len(w)
        while st < N:
            ed = min(st + mx, N)
            f = ed // fl
            while f < len(sp) and sp[f]:
                f += 1; ed = min(f * fl, N)
                if ed - st > mx * 1.5: break
            if ed - st < mn and ed < N: ed = min(st + mn, N)
            cs.append(wav[:, st:ed].squeeze())
            ts.append([round(st / sr, 2), round(ed / sr, 2)])
            st = ed
        return cs, torch.tensor(ts)



def padding_audio(batch):
    audios, times = zip(*batch)
    return pad_sequence(audios, batch_first=True), torch.tensor([audio.numel() for audio in audios]), torch.stack(times)

class Op(nn.Module):
    def __init__(self, func,allow_self=False):
        super().__init__()
        self.func = func
        self.allow_self = allow_self

    def forward(self, x):
        if self.allow_self:
            return self.func(self,x)
        return self.func(x)

class Wav2Vec2ConformerRNNT(Wav2Vec2ConformerModel):
    def __init__(self, config):
        self.language = config.languages[0]
        if len(config.languages) > 1:
            config.hidden_size = 1024
            config.num_hidden_layers = 24
            config.conv_depthwise_kernel_size = 9
            config.conv_stride = [2,2,2]
            config.conv_kernel = [3,3,3]
            config.conv_dim = [256,256,256]
            config.feat_extract_norm = "group"
            config.intermediate_size = 4096
            config.num_feat_extract_layers = len(config.conv_dim)
            config.lstm_layer = 2

        self.cache_length = None
        self.hop, self.preemph, self.eps, self.pad_to = 160, 0.97, 2**-24, 16
        self.denorm = (2 ** config.num_feat_extract_layers) * self.hop / config.sampling_rate
        self.scaler = config.hidden_size ** (1/2)
        super().__init__(config)
        self.eval()

    def init_weights(self):
        del self.encoder.pos_conv_embed
        config = self.config
        self.enc = nn.Linear(config.hidden_size, config.joint_hidden)
        self.pred = nn.Linear(config.pred_hidden, config.joint_hidden)
        self.joint = nn.Linear(config.joint_hidden, config.vocab_size // 22 + 1)
        self.embed = nn.Embedding(config.vocab_size+1, config.pred_hidden, padding_idx=config.vocab_size)
        self.lstm = nn.LSTM(config.pred_hidden, config.pred_hidden, config.lstm_layer, batch_first=True)
        self.act = nn.ReLU(inplace=True)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=400, center=False)
        self.mask_layer = Op(lambda self_obj,x : x.masked_fill(self_obj.cache_pad_mask.unsqueeze(1), 0),True)
        self.mel_fb = nn.Parameter(torch.tensor(librosa.filters.mel(sr=self.config.sampling_rate, n_fft=512, n_mels=80)),False)

        for idx,l in enumerate(self.feature_extractor.conv_layers):
            if len(self.config.languages) == 1 or idx == 0:
                l.conv = nn.Conv2d(l.conv.in_channels,l.conv.out_channels,l.conv.kernel_size[0],l.conv.stride,1)
                l.layer_norm = nn.Identity()
            else:
                l.conv = nn.Sequential(nn.Conv2d(l.conv.in_channels,l.conv.out_channels,l.conv.kernel_size[0],l.conv.stride,1,groups=l.conv.out_channels),nn.Conv2d(l.conv.in_channels,l.conv.out_channels, 1))

        self.feature_extractor.conv_layers.append(Op(lambda x : x.transpose(1, 2)))
        self.feature_projection.projection = nn.Linear(config.conv_dim[-1] * int(calc_length(torch.tensor(80.),repeat_num=self.config.num_feat_extract_layers)),config.hidden_size)
        self.feature_projection.layer_norm = Op(lambda x:x.permute(0, 2, 1, 3).flatten(2))
        for l in self.encoder.layers:
            l.conv_module.glu = nn.Sequential(l.conv_module.glu,self.mask_layer)
            l.conv_module.pointwise_conv1.bias = nn.Parameter(torch.empty(l.conv_module.pointwise_conv1.out_channels))
            l.conv_module.pointwise_conv2.bias = nn.Parameter(torch.empty(l.conv_module.pointwise_conv2.out_channels))
            l.conv_module.depthwise_conv.bias = nn.Parameter(torch.empty(l.conv_module.depthwise_conv.out_channels))
        self.encoder.layer_norm = nn.Identity()
        if len(self.config.languages) > 1:
            self.lang_joint_net = nn.ModuleDict({l: nn.Linear(config.joint_hidden, config.vocab_size // 22 + 1) for l in config.languages})
        return super().init_weights()

    def _mask_hidden_states(self, hidden_states, mask_time_indices = None, attention_mask = None):
        hidden_states = hidden_states * self.scaler
        self.mask_layer.cache_pad_mask = (torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0) >= self.cache_length.unsqueeze(1))
        return super()._mask_hidden_states(hidden_states, mask_time_indices, attention_mask)

    def preprocessing(self, x):
        x, l = x
        l = (l // self.hop + 1).long()
        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), 1)
        x = (self.mel_fb @ self.spec(x) + self.eps).log()
        T = x.size(-1)
        m = torch.arange(T, device=x.device)[None] >= l[:, None]
        x = x.masked_fill(m[:, None], 0)
        μ = x.sum(-1) / l[:, None]
        σ = (((x - μ[..., None])**2).sum(-1) / (l[:, None] - 1) + 1e-5).sqrt()
        x = ((x - μ[..., None]) / σ[..., None]).masked_fill(m[:, None], 0)
        self.cache_length = calc_length(l, repeat_num=self.config.num_feat_extract_layers).long()
        return F.pad(x, (0, (-T) % self.pad_to)).transpose(1, 2)

    def forward(self, input_values):
        return self.postprocessing(super().forward(self.preprocessing(input_values)).last_hidden_state)

    @torch.inference_mode()
    def transcribe(self,wav,sr,batch_size):
        device = next(self.parameters()).device
        subtitles = []
        for batch, lengths, timestamp in DataLoader(ChunkedData(wav, sr),batch_size,collate_fn=padding_audio):
            batch = batch.to(device)
            lengths = lengths.to(device)
            timestamp = timestamp.to(device)
            subtitles.extend(self.make_srt(self.forward((batch, lengths)),timestamp))
            yield srt.compose(subtitles)
            torch.cuda.empty_cache()
            gc.collect()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        del state_dict['ctc_decoder.decoder_layers.0.bias']
        del state_dict['ctc_decoder.decoder_layers.0.weight']
        state_dict['preprocessor.featurizer.fb'] = state_dict['preprocessor.featurizer.fb'].squeeze(0)
        changes = """
preprocessor.featurizer.fb,mel_fb
preprocessor.featurizer.window,spec.window
norm_feed_forward1,ffn1_layer_norm
norm_feed_forward2,ffn2_layer_norm
feed_forward1.linear1,ffn1.intermediate_dense
feed_forward1.linear2,ffn1.output_dense
feed_forward2.linear1,ffn2.intermediate_dense
feed_forward2.linear2,ffn2.output_dense
norm_self_att,self_attn_layer_norm
norm_out,final_layer_norm
norm_conv,conv_module.layer_norm
.conv.,.conv_module.
decoder.prediction.dec_rnn.lstm,lstm
decoder.prediction.embed,embed
joint.enc,enc
joint.pred,pred
joint.joint_net.2,lang_joint_net
encoder.pre_encode.conv_module.0,feature_extractor.conv_layers.0.conv
encoder.pre_encode.out,feature_projection.projection
"""
        if len(self.config.languages) == 1:
            changes += f"""lang_joint_net.{self.language},joint
encoder.pre_encode.conv_module.{{n}},feature_extractor.conv_layers.{{(n/2)}}.conv"""
        else:
            state_dict["joint.weight"] = self.joint.weight.clone()
            state_dict["joint.bias"] = self.joint.bias.clone()
            changes += """encoder.pre_encode.conv_module.{n},encoder.pre_encode.conv_module.{(n-2)}
encoder.pre_encode.conv_module.{n},feature_extractor.conv_layers.{(n//3+1)}.conv.{(n%3)}
"""
            # replicate many changes for complex maths
        state_dict = state_bridge(state_dict, changes)
        if len(self.config.languages) == 1:
            state_dict = {k: v for k, v in state_dict.items() if "lang_joint_net" not in k}
        return super().load_state_dict(state_dict, strict, assign)

    def postprocessing(self, x):
        if len(self.config.languages) > 1:
            self.joint.load_state_dict(self.lang_joint_net[self.language].state_dict())
        B = x.size(0)
        last = x.new_full((B, 1), self.config.blank_id, dtype=torch.long)
        h, tok, st = None, [[] for _ in range(B)], [[] for _ in range(B)]
        for t, e in enumerate(x.unbind(1)):
            v = t < self.cache_length
            if not v.any(): break
            e = e[:, None]
            for _ in range(self.config.max_symbols_per_step):
                p, h2 = self.lstm(self.embed(last), h)
                lg = self.joint(self.act(self.enc(e) + self.pred(p))).squeeze(1)
                n = torch.where(v, lg.argmax(-1), self.config.blank_id)
                b = n.eq(self.config.blank_id)
                if b.all(): break
                a = v & ~b
                for i in a.nonzero().flatten().tolist():
                    tok[i].append(n[i]); st[i].append(t * self.denorm)
                last = torch.where(a[:, None], n[:, None], last)
                if h is None: h = h2
                else:
                    k = (b | ~v).view(1, -1, 1)
                    h = (torch.where(k, h[0], h2[0]), torch.where(k, h[1], h2[1]))
        self.cache_length = None
        return torch.tensor(tok), torch.tensor(st)

    def make_srt(self, x, ts):
        t , s = x
        start_token_segment = self.config.languages.index(self.language) * self.joint.out_features
        all_tokens, all_starts, all_ends = [], [], []
        for tokens, starts, (s, e) in zip(t,s, ts):
            tokens += start_token_segment
            starts += s
            all_tokens.append(tokens)
            all_starts.append(starts)
            all_ends.append(torch.cat([starts[1:], e[None]]))
            all_tokens.append(torch.tensor([-1]))
            all_starts.append(torch.tensor([e]))
            all_ends.append(torch.tensor([e + 0.005]))
        return [srt.Subtitle(i,timedelta(seconds=float(st)),timedelta(seconds=float(en)),"<line>" if tok == -1 else self.config.vocab[int(tok)]) for i, (tok, st, en) in enumerate(zip(torch.cat(all_tokens), torch.cat(all_starts), torch.cat(all_ends)), 1)]


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config = None, language=None,**kwargs):
        if language:
            config.languages = [language]
            config.vocab = ['<unk>'] + json.load(open(hf_hub_download(pretrained_model_name_or_path, "vocab.json")))['small'][language]
        else:
            temp_vocab = json.load(open(hf_hub_download(pretrained_model_name_or_path, "vocab.json")))['large']
            config.vocab = []
            for i in sorted(config.languages):
                config.vocab.extend(['<unk>'] + temp_vocab[i])
        model = cls(config)
        model.load_state_dict(load_file(hf_hub_download(pretrained_model_name_or_path, f"{language or 'all'}.safetensors")))
        return model
