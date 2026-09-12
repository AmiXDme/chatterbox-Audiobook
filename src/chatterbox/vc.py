from pathlib import Path

import librosa
import torch
from huggingface_hub import hf_hub_download

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"

# Local models directory inside the project folder
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        import time as _lt
        ckpt_dir = Path(ckpt_dir)
        print(f"[VC] Loading weights from {ckpt_dir} ...", flush=True)
        _w0 = _lt.time()
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=torch.device('cpu'))
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=torch.device('cpu'))
        )
        s3gen.to(device).eval()
        print(f"[VC] Weights ready in {_lt.time()-_w0:.1f}s", flush=True)

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Download models into the project's models/ folder
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_files = ["s3gen.pt", "conds.pt"]
        missing = [f for f in model_files if not (_MODELS_DIR / f).exists()]
        print(f"[VC] Checking {len(model_files)} files in {_MODELS_DIR} | missing={missing or 'none'}", flush=True)

        for fpath in model_files:
            dest = _MODELS_DIR / fpath
            if not dest.exists():
                print(f"  Downloading {fpath} (large file, terminal quiet until done)...", flush=True)
                hf_hub_download(repo_id=REPO_ID, filename=fpath, local_dir=str(_MODELS_DIR))
                print(f"  Downloaded: {fpath}", flush=True)

        return cls.from_local(_MODELS_DIR, device)

    @torch.inference_mode()
    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)

        if self.ref_dict is None:
            raise ValueError("Please `prepare_conditionals` first or specify `target_voice_path`")

        with torch.inference_mode():
            import time as _vt
            print("🎙️ [VC] Tokenizing source audio...", flush=True)
            _v0 = _vt.time()
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            _n_vtok = int(s3_tokens.shape[-1])
            _vt_dur = _vt.time() - _v0
            print(f"🎙️ [VC] Tokenized {audio_16.shape[-1]/S3_SR:.1f}s -> {_n_vtok} tokens in {_vt_dur:.1f}s ({_n_vtok/max(_vt_dur,1e-3):.1f} tok/s)", flush=True)
            print(f"🎻 [VC] Decoding {_n_vtok} tokens with S3Gen...", flush=True)
            _d0 = _vt.time()
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.ref_dict,
            )
            _d_dur = _vt.time() - _d0
            print(f"🔊 [VC] Decode done in {_d_dur:.1f}s ({_n_vtok/max(_d_dur,1e-3):.1f} tok/s) -> {wav.shape[-1]/S3GEN_SR:.1f}s audio", flush=True)
            wav = wav.squeeze(0).detach().cpu().numpy()
        print("✅ [VC] Voice conversion complete.", flush=True)
        return torch.from_numpy(wav).unsqueeze(0)
