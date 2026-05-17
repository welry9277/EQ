from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import yaml

from .base import BaseClapModel


def _resolve_path(path_str: str | None) -> Path:
    if not path_str:
        raise ValueError("checkpoint_path is required for MGA-CLAP.")
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


class MGAClapModel(BaseClapModel):
    """MGA-CLAP (Ming-er/MGA-CLAP) via ``models.ase_model.ASE`` and a settings YAML."""

    def _load_model(self) -> None:
        if not self.repo_path:
            raise ValueError(
                "MGA-CLAP requires repo_path pointing to a clone of Ming-er/MGA-CLAP."
            )
        repo = Path(self.repo_path).resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"MGA-CLAP repo_path not found: {repo}")

        root = str(repo)
        if root not in sys.path:
            sys.path.insert(0, root)

        settings_rel = self.config.get("settings_yaml", "settings/inference_example.yaml")
        settings_path = (repo / settings_rel).resolve()
        if not settings_path.is_file():
            raise FileNotFoundError(
                f"MGA settings YAML not found: {settings_path} "
                f"(set 'settings_yaml' in model config relative to repo_path)."
            )

        ckpt_path = _resolve_path(self.checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MGA checkpoint not found: {ckpt_path}")

        with open(settings_path, "r", encoding="utf-8") as handle:
            mga_cfg = yaml.safe_load(handle)
        mga_cfg = copy.deepcopy(mga_cfg)
        mga_cfg["device"] = str(self.device)
        mga_cfg.setdefault("eval", {})["ckpt"] = str(ckpt_path)

        try:
            from models.ase_model import ASE
        except ImportError as exc:
            raise ImportError(
                "Cannot import models.ase_model. Clone https://github.com/Ming-er/MGA-CLAP "
                f"into {repo} and install its dependencies."
            ) from exc

        print(f"Loading MGA-CLAP from {ckpt_path} on {self.device}")
        self.model = ASE(mga_cfg)
        try:
            state = torch.load(
                str(ckpt_path), map_location=self.device, weights_only=False
            )
        except TypeError:
            state = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self._mga_cfg = mga_cfg
        self.target_sr = int(mga_cfg["audio_args"]["sr"])
        self.max_audio_seconds = float(mga_cfg["audio_args"].get("max_length", 10))
        self.max_samples = int(self.target_sr * self.max_audio_seconds)
        self.resampler_cache: dict[int, T.Resample] = {}

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: list[np.ndarray], sr: int) -> np.ndarray:
        batch = []
        for arr in audio_data:
            if arr.ndim > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            wav = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)

            if sr != self.target_sr:
                if sr not in self.resampler_cache:
                    self.resampler_cache[sr] = T.Resample(
                        orig_freq=sr, new_freq=self.target_sr
                    ).to(self.device)
                wav = wav.to(self.device)
                wav = self.resampler_cache[sr](wav)
            else:
                wav = wav.to(self.device)

            if wav.shape[1] > self.max_samples:
                wav = wav[:, : self.max_samples]
            elif wav.shape[1] < self.max_samples:
                wav = F.pad(wav, (0, self.max_samples - wav.shape[1]))

            batch.append(wav)

        wav_b = torch.cat(batch, dim=0)
        _, frame_embeds = self.model.encode_audio(wav_b)
        audio_embeds = self.model.msc(frame_embeds, self.model.codebook)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        _, word_embeds, attn_mask = self.model.encode_text(texts)
        text_embeds = self.model.msc(word_embeds, self.model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds.cpu().numpy()
