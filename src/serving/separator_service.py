from pathlib import Path
import os

import hydra
import soundfile as sf
from omegaconf import OmegaConf

from src.utils.utils import load_wav
from src.evaluation.separate import separate_with_ckpt_TDF
from typing import Union, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class StemSeparator:
    def __init__(
        self,
        ckpt_map: dict,
        device: str = "cuda:0",
        batch_size: int = 1,
        double_chunk: bool = False,
        overlap_add: Union[dict, None] = None,  # 或 Optional[dict]
        samplerate: int = 44100,
    ):
        self.ckpt_map = ckpt_map
        self.device = device
        self.batch_size = batch_size
        self.double_chunk = double_chunk
        self.overlap_add = overlap_add
        self.samplerate = samplerate
        self.model_cache = {}

    def _load_model_cfg(self, stem: str):
        if stem in self.model_cache:
            return self.model_cache[stem]

        cfg_path = PROJECT_ROOT / "configs" / "model" / f"{stem}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"找不到模型配置文件: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(cfg)

        self.model_cache[stem] = model
        return model

    def _build_overlap_cfg(self):
        if self.overlap_add is None:
            return None

        class OverlapCfg:
            pass

        obj = OverlapCfg()
        obj.overlap_rate = self.overlap_add["overlap_rate"]
        obj.tmp_root = self.overlap_add["tmp_root"]
        obj.samplerate = self.overlap_add["samplerate"]
        return obj

    def separate_one(self, audio_path: str, stem: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)

        model = self._load_model_cfg(stem)
        ckpt_path = Path(self.ckpt_map[stem])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

        mixture = load_wav(audio_path)

        target_hat = separate_with_ckpt_TDF(
            self.batch_size,
            model,
            ckpt_path,
            mixture,
            self.device,
            self.double_chunk,
            self._build_overlap_cfg(),
        )

        out_path = Path(out_dir) / f"{stem}.wav"
        sf.write(out_path, target_hat.T, self.samplerate)
        return str(out_path)

    def separate_all(self, audio_path: str, out_dir: str) -> dict:
        result = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            result[stem] = self.separate_one(audio_path, stem, out_dir)
        return result