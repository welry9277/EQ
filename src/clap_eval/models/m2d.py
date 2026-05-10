from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from .base import BaseClapModel


def _resolve_path(path_str: str | None) -> Path:
    if not path_str:
        raise ValueError("checkpoint_path is required for M2D.")
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _m2d_weight_file(repo_path: Path, config: dict, checkpoint_path: Path) -> Path:
    """PortableM2D infers architecture from the checkpoint file's parent directory name."""
    if checkpoint_path.is_file():
        parent = checkpoint_path.parent.name.lower()
        if "m2d" in parent and "x" in checkpoint_path.parent.name:
            return checkpoint_path

    folder = config.get("m2d_weight_folder_name")
    if not folder:
        raise ValueError(
            "M2D: place checkpoint under a folder named like "
            "'m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025' (see nttcslab/m2d releases), "
            "or set 'm2d_weight_folder_name' in the model config to that folder name "
            "under repo_path."
        )
    candidate = (repo_path / folder / checkpoint_path.name).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"M2D checkpoint not found at {candidate}. "
            f"Clone nttcslab/m2d into repo_path and extract weights there."
        )
    return candidate


class M2DClapModel(BaseClapModel):
    """M2D-CLAP via nttcslab/m2d ``examples/portable_m2d.PortableM2D``."""

    def _load_model(self) -> None:
        if not self.repo_path:
            raise ValueError("M2D requires repo_path pointing to a clone of nttcslab/m2d.")
        repo = Path(self.repo_path).resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"M2D repo_path not found: {repo}")

        root = str(repo)
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from examples.portable_m2d import PortableM2D
        except ImportError as exc:
            raise ImportError(
                "Cannot import examples.portable_m2d. Clone https://github.com/nttcslab/m2d "
                f"into {repo} and install deps (timm, einops, nnAudio, transformers, …)."
            ) from exc

        ckpt_arg = _resolve_path(self.checkpoint_path)
        weight_file = _m2d_weight_file(repo, self.config, ckpt_arg)

        print(f"Loading M2D-CLAP from {weight_file} on {self.device}")
        self.model = PortableM2D(str(weight_file))
        self.model = self.model.to(self.device).eval()

        self.target_sr = int(self.model.cfg.sample_rate)
        self.max_audio_samples = int(
            self.model.cfg.input_size[1] * self.model.cfg.hop_size
        )
        self.resampler_cache: dict[int, T.Resample] = {}

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        processed: list[np.ndarray] = []
        for arr in audio_data:
            if arr.ndim > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            processed.append(np.asarray(arr, dtype=np.float32))

        max_len = max(len(a) for a in processed) if processed else 0
        padded = np.zeros((len(processed), max_len), dtype=np.float32)
        for i, a in enumerate(processed):
            padded[i, : len(a)] = a

        wav = torch.from_numpy(padded).float()
        if sr != self.target_sr:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = T.Resample(
                    orig_freq=sr, new_freq=self.target_sr
                ).to(self.device)
            wav = wav.to(self.device)
            wav = self.resampler_cache[sr](wav)
        else:
            wav = wav.to(self.device)

        if wav.shape[-1] > self.max_audio_samples:
            wav = wav[..., : self.max_audio_samples]
        elif wav.shape[-1] < self.max_audio_samples:
            wav = F.pad(wav, (0, self.max_audio_samples - wav.shape[-1]))

        emb = self.model.encode_clap_audio(wav)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode_clap_text(texts)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb, device=self.device)
        emb = F.normalize(emb.float(), p=2, dim=-1)
        return emb.cpu().numpy()
