import os
import random
import soundfile as sf

INPUT_WAV = r"E:\2232813\projects\DTTNET\audio\mixture.wav"
OUTPUT_DIR = r"E:\2232813\projects\DTTNET\audio"

CLIP_SECONDS = 8
NUM_CLIPS = 5   # 想切几个就改这里
SEED = None     # 想固定随机结果就改成整数，比如 42


def main():
    if SEED is not None:
        random.seed(SEED)

    if not os.path.exists(INPUT_WAV):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_WAV}")

    audio, sr = sf.read(INPUT_WAV)
    total_samples = len(audio)
    clip_samples = CLIP_SECONDS * sr

    if total_samples < clip_samples:
        raise ValueError(f"音频总时长不足 {CLIP_SECONDS} 秒，无法切分。")

    max_start = total_samples - clip_samples

    if NUM_CLIPS > max_start + 1:
        raise ValueError("可选起点数量不足，NUM_CLIPS 设得太大。")

    starts = random.sample(range(max_start + 1), NUM_CLIPS)
    starts.sort()

    base_name = os.path.splitext(os.path.basename(INPUT_WAV))[0]

    print("开始切分...")
    for i, start in enumerate(starts, 1):
        end = start + clip_samples
        clip_audio = audio[start:end]

        start_sec = start / sr
        out_name = f"{base_name}_random_{i:02d}_{start_sec:.2f}s.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        sf.write(out_path, clip_audio, sr)
        print(f"已保存: {out_path}")

    print("全部完成。")


if __name__ == "__main__":
    main()