from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.serving.separator_service import StemSeparator


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(config: DictConfig):
    stem = config.model.target_name

    overlap_add = None
    if config.get("overlap_add") is not None:
        overlap_add = OmegaConf.to_container(config.overlap_add, resolve=True)

        # 把 tmp_root 也转成绝对路径，避免 Hydra 切 cwd 后跑偏
        if overlap_add.get("tmp_root") is not None:
            overlap_add["tmp_root"] = to_absolute_path(overlap_add["tmp_root"])

    separator = StemSeparator(
        ckpt_map={
            stem: to_absolute_path(config.ckpt_path)
        },
        device=config.device,
        batch_size=config.batch_size,
        double_chunk=config.double_chunk,
        overlap_add=overlap_add,
        samplerate=config.samplerate,
    )

    mixture_path = to_absolute_path(config.mixture_path)

    # 输出目录：项目根目录/infer/<stem>/
    out_dir = Path(to_absolute_path("infer")) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = separator.separate_one(
        audio_path=mixture_path,
        stem=stem,
        out_dir=str(out_dir),
    )

    print(f"[OK] separated stem saved to: {out_path}")


if __name__ == "__main__":
    main()