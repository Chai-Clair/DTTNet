from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
import librosa
import soundfile as sf
import json

from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import AUDIO_SAMPLE_RATE

from music21 import converter, stream, note, meter, clef, tempo, instrument


def midi_to_musicxml(midi_path: str, xml_path: str):
    score = converter.parse(midi_path)
    score.write("musicxml", fp=xml_path)
    return xml_path

def _find_musescore_bin() -> str:
    env_bin = os.getenv("MUSESCORE_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin

    # Windows 本地固定安装路径优先
    windows_candidates = [
        Path(r"F:\MuseScore 3\bin\MuseScore3.exe"),
        Path(r"F:\MuseScore 3\bin\MuseScore.exe"),
    ]
    for p in windows_candidates:
        if p.exists():
            return str(p)

    candidates = [
        "musescore",
        "mscore",
        "mscore3",
        "musescore3",
        "MuseScore4",
        "MuseScore3",
    ]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path

    raise FileNotFoundError(
        "未找到 MuseScore 命令。Windows 可检查 F:\\MuseScore 3\\bin，"
        "或设置环境变量 MUSESCORE_BIN。"
    )
def musicxml_to_pdf(xml_path: str, pdf_path: str) -> str:
    musescore_bin = _find_musescore_bin()

    cmd = [
        musescore_bin,
        xml_path,
        "-o",
        pdf_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pdf_path

def _transcribe_basic_pitch_stem(wav_path: str, out_prefix: str, stem_name: str) -> dict:
    out_dir = Path(out_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_wav = _prepare_basic_pitch_input(wav_path, out_dir, stem_name)

    predict_and_save(
        audio_path_list=[str(prepared_wav)],
        output_directory=str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    midi_files = sorted(out_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not midi_files:
        print(f"[WARN] {stem_name} 没有生成 mid，目录: {out_dir}")
        print(f"[WARN] 当前目录文件: {[p.name for p in out_dir.iterdir()]}")
        return {"midi": None, "musicxml": None, "pdf": None}

    midi_path = midi_files[0]
    xml_path = out_dir / f"{stem_name}.musicxml"
    pdf_path = out_dir / f"{stem_name}.pdf"

    midi_to_musicxml(str(midi_path), str(xml_path))
    musicxml_to_pdf(str(xml_path), str(pdf_path))

    return {
        "midi": str(midi_path),
        "musicxml": str(xml_path),
        "pdf": str(pdf_path),
    }

def _prepare_basic_pitch_input(wav_path: str, out_dir: Path, stem_name: str) -> Path:
    """
    先把输入音频转成 basic_pitch 目标格式：
    - mono
    - sample rate = AUDIO_SAMPLE_RATE
    - wav
    避免 basic_pitch 内部再走 librosa/resampy 重采样链。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_wav = out_dir / f"{stem_name}_input.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        str(prepared_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return prepared_wav

VOCAL_HOP_LENGTH = 256
VOCAL_MIN_MIDI = librosa.note_to_midi("A2")
VOCAL_MAX_MIDI = librosa.note_to_midi("C6")

VOCAL_MIN_VOICED_PROB = 0.30
VOCAL_MAX_SHORT_GAP_FRAMES = 6
VOCAL_MIN_NOTE_SEC = 0.10
VOCAL_SPLIT_JUMP_SEMITONES = 2.0

VOCAL_DURATION_CANDIDATES_QL = np.array(
    [0.25, 1/3, 0.5, 2/3, 1.0, 1.5, 2.0, 3.0, 4.0],
    dtype=float
)


def _load_mono_no_resample(path: str):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32, copy=False)
    return y, sr


def _estimate_tempo_from_mixture(mixture_wav_path: str):
    y, sr = _load_mono_no_resample(mixture_wav_path)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=VOCAL_HOP_LENGTH)
    tempo_est, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=VOCAL_HOP_LENGTH,
        units="frames"
    )

    tempo_val = float(np.atleast_1d(tempo_est)[0])
    if not np.isfinite(tempo_val) or tempo_val < 40 or tempo_val > 220:
        tempo_val = 120.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=VOCAL_HOP_LENGTH)
    return tempo_val, beat_times


def _quantize_offset_ql(offset_sec: float, beat_sec: float) -> float:
    raw_ql = offset_sec / beat_sec
    # 1/12 拍网格，能兼容 16 分音符(1/4) 和八分三连音(1/3)
    return round(raw_ql * 12) / 12.0


def _quantize_duration_ql(duration_sec: float, beat_sec: float) -> float:
    raw_ql = max(duration_sec / beat_sec, 1/12)
    idx = int(np.argmin(np.abs(VOCAL_DURATION_CANDIDATES_QL - raw_ql)))
    return float(VOCAL_DURATION_CANDIDATES_QL[idx])

def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue

        j = i
        while j < n and not mask[j]:
            j += 1

        gap_len = j - i
        left_on = i > 0 and mask[i - 1]
        right_on = j < n and mask[j]

        if left_on and right_on and gap_len <= max_gap:
            mask[i:j] = True

        i = j

    return mask


def _interpolate_short_nan_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    x = values.copy()
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return x

    idx = np.arange(len(x))
    i = 0
    while i < len(x):
        if np.isfinite(x[i]):
            i += 1
            continue

        j = i
        while j < len(x) and not np.isfinite(x[j]):
            j += 1

        gap_len = j - i
        left_ok = i > 0 and np.isfinite(x[i - 1])
        right_ok = j < len(x) and np.isfinite(x[j])

        if left_ok and right_ok and gap_len <= max_gap:
            x[i:j] = np.interp(idx[i:j], [i - 1, j], [x[i - 1], x[j]])

        i = j

    return x

def _segment_monophonic_vocal(y: np.ndarray, sr: int):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.midi_to_hz(VOCAL_MIN_MIDI),
        fmax=librosa.midi_to_hz(VOCAL_MAX_MIDI),
        sr=sr,
        hop_length=VOCAL_HOP_LENGTH,
        frame_length=2048,
    )

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=VOCAL_HOP_LENGTH,
        backtrack=False,
        units="frames"
    )
    onset_set = set(int(x) for x in onset_frames)

    midi_track = librosa.hz_to_midi(f0)
    midi_track = _interpolate_short_nan_gaps(
        midi_track,
        max_gap=VOCAL_MAX_SHORT_GAP_FRAMES
    )

    active_mask = (
        np.isfinite(midi_track)
        & (
            np.asarray(voiced_flag, dtype=bool)
            | (np.nan_to_num(voiced_prob, nan=0.0) >= VOCAL_MIN_VOICED_PROB)
        )
    )

    active_mask = _fill_short_false_gaps(
        active_mask,
        max_gap=VOCAL_MAX_SHORT_GAP_FRAMES
    )

    frame_times = librosa.frames_to_time(
        np.arange(len(midi_track)),
        sr=sr,
        hop_length=VOCAL_HOP_LENGTH
    )

    segments = []
    n = len(midi_track)
    i = 0

    while i < n:
        if not active_mask[i]:
            i += 1
            continue

        j = i
        while j + 1 < n and active_mask[j + 1]:
            j += 1

        split_points = [i]

        for k in range(i + 1, j + 1):
            if not np.isfinite(midi_track[k]) or not np.isfinite(midi_track[k - 1]):
                continue

            jump = abs(float(midi_track[k]) - float(midi_track[k - 1]))

            if k in onset_set and (k - split_points[-1]) >= 4:
                split_points.append(k)
            elif jump >= VOCAL_SPLIT_JUMP_SEMITONES and (k - split_points[-1]) >= 4:
                split_points.append(k)

        split_points.append(j + 1)

        for a, b in zip(split_points[:-1], split_points[1:]):
            vals = midi_track[a:b]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue

            start_t = float(frame_times[a])
            end_t = float(frame_times[b - 1] + VOCAL_HOP_LENGTH / sr)

            if end_t - start_t < VOCAL_MIN_NOTE_SEC:
                continue

            pitch_midi = float(np.median(vals))
            segments.append({
                "start_sec": start_t,
                "end_sec": end_t,
                "midi": pitch_midi,
            })

        i = j + 1

    return segments

def _merge_same_pitch_neighbors(segments):
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        same_pitch = abs(seg["midi"] - prev["midi"]) < 0.75
        small_gap = (seg["start_sec"] - prev["end_sec"]) < 0.16

        if same_pitch and small_gap:
            prev["end_sec"] = seg["end_sec"]
        else:
            merged.append(seg)

    return merged

def _unwrap_vocal_octaves_by_continuity(segments):
    if not segments:
        return segments

    # 先找一个锚点：最长的音，通常更可信
    anchor_idx = max(
        range(len(segments)),
        key=lambda i: segments[i]["end_sec"] - segments[i]["start_sec"]
    )

    stabilized = [dict(seg) for seg in segments]
    stabilized[anchor_idx]["midi"] = float(stabilized[anchor_idx]["midi"])

    def best_octave(raw_midi, ref_midi):
        candidates = []
        for shift in (-24, -12, 0, 12, 24):
            cand = float(raw_midi + shift)
            # 只保留比较合理的人声音域
            if 48 <= cand <= 84:
                candidates.append(cand)

        if not candidates:
            return float(raw_midi)

        # 选离参考音最近的那个八度版本
        return min(candidates, key=lambda x: abs(x - ref_midi))

    # 向右展开
    for i in range(anchor_idx + 1, len(stabilized)):
        prev_midi = float(stabilized[i - 1]["midi"])
        raw_midi = float(stabilized[i]["midi"])
        stabilized[i]["midi"] = best_octave(raw_midi, prev_midi)

    # 向左展开
    for i in range(anchor_idx - 1, -1, -1):
        next_midi = float(stabilized[i + 1]["midi"])
        raw_midi = float(stabilized[i]["midi"])
        stabilized[i]["midi"] = best_octave(raw_midi, next_midi)

    # 再做一次极短孤立音清理
    cleaned = []
    for i, seg in enumerate(stabilized):
        dur = seg["end_sec"] - seg["start_sec"]
        midi_val = seg["midi"]

        prev_midi = stabilized[i - 1]["midi"] if i > 0 else None
        next_midi = stabilized[i + 1]["midi"] if i + 1 < len(stabilized) else None

        isolated = False
        if prev_midi is not None and next_midi is not None:
            if abs(midi_val - prev_midi) > 8 and abs(midi_val - next_midi) > 8:
                isolated = True

        if dur < 0.14 and isolated:
            continue

        cleaned.append(seg)

    return cleaned

def transcribe_vocals(vocal_wav_path: str, mixture_wav_path: str, out_prefix: str) -> dict:
    out_dir = Path(out_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 用 mixture 估计全局 BPM
    bpm, beat_times = _estimate_tempo_from_mixture(mixture_wav_path)
    beat_sec = 60.0 / bpm

    # 2) 在 vocals 上做单声部音高跟踪
    y, sr = _load_mono_no_resample(vocal_wav_path)
    segments = _segment_monophonic_vocal(y, sr)
    segments = _merge_same_pitch_neighbors(segments)
    segments = _unwrap_vocal_octaves_by_continuity(segments)
    segments = _merge_same_pitch_neighbors(segments)

    debug_path = out_dir / "vocals_segments.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "dur_sec": float(seg["end_sec"] - seg["start_sec"]),
                    "midi": float(seg["midi"]),
                }
                for seg in segments
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    if not segments:
        return {"midi": None, "musicxml": None, "pdf": None}

    # 3) 构建单旋律谱
    sc = stream.Score()
    pt = stream.Part()
    pt.append(instrument.Vocalist())
    pt.append(meter.TimeSignature("4/4"))
    pt.append(tempo.MetronomeMark(number=round(bpm)))

    # 先用“第一音开始时间”为 0，版本1先做 BPM-aware 量化，不做严格小节对齐
    anchor_sec = segments[0]["start_sec"]

    quantized_notes = []

    for seg in segments:
        start_sec = max(0.0, seg["start_sec"] - anchor_sec)
        dur_sec = max(0.06, seg["end_sec"] - seg["start_sec"])

        offset_ql = _quantize_offset_ql(start_sec, beat_sec)
        dur_ql = _quantize_duration_ql(dur_sec, beat_sec)
        midi_val = int(round(seg["midi"]))

        quantized_notes.append({
            "start_sec": float(start_sec),
            "dur_sec": float(dur_sec),
            "offset_ql": float(offset_ql),
            "dur_ql": float(dur_ql),
            "midi": int(midi_val),
        })

        n = note.Note()
        n.pitch.midi = midi_val
        n.quarterLength = dur_ql
        pt.insert(offset_ql, n)

    quantized_path = out_dir / "vocals_quantized.json"
    with open(quantized_path, "w", encoding="utf-8") as f:
        json.dump(quantized_notes, f, ensure_ascii=False, indent=2)

    pt.makeMeasures(inPlace=True)
    sc.insert(0, pt)

    midi_path = out_dir / "vocals.mid"
    xml_path = out_dir / "vocals.musicxml"
    pdf_path = out_dir / "vocals.pdf"

    sc.write("midi", fp=str(midi_path))
    sc.write("musicxml", fp=str(xml_path))
    musicxml_to_pdf(str(xml_path), str(pdf_path))

    return {
        "midi": str(midi_path),
        "musicxml": str(xml_path),
        "pdf": str(pdf_path),
    }

def transcribe_bass(wav_path: str, out_prefix: str) -> dict:
    return _transcribe_basic_pitch_stem(wav_path, out_prefix, "bass")

def transcribe_drums(wav_path: str, out_prefix: str) -> dict:
    out_dir = Path(out_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = sf.read(wav_path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32, copy=False)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=False
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    sc = stream.Score()
    pt = stream.Part()
    pt.append(clef.PercussionClef())
    pt.append(meter.TimeSignature("4/4"))
    pt.append(tempo.MetronomeMark(number=120))

    ql_per_sec = 120.0 / 60.0
    grid = 0.25
    used_offsets = set()

    for t in onset_times:
        idx = int(t * sr)
        seg = y[idx:min(idx + int(0.08 * sr), len(y))]
        if len(seg) < 32:
            continue

        S = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

        low_mask = (freqs >= 20) & (freqs < 180)
        mid_mask = (freqs >= 180) & (freqs < 1200)
        high_mask = (freqs >= 1200)

        low = S[low_mask].mean() if np.any(low_mask) else 0.0
        mid = S[mid_mask].mean() if np.any(mid_mask) else 0.0
        high = S[high_mask].mean() if np.any(high_mask) else 0.0

        n = note.Unpitched()

        if low >= mid and low >= high:
            n.storedInstrument = instrument.BassDrum()
            n.displayStep = "F"
            n.displayOctave = 4
        elif mid >= low and mid >= high:
            n.storedInstrument = instrument.SnareDrum()
            n.displayStep = "C"
            n.displayOctave = 5
        else:
            n.storedInstrument = instrument.HiHatCymbal()
            n.displayStep = "G"
            n.displayOctave = 5

        n.quarterLength = grid

        offset_ql = t * ql_per_sec
        offset_ql = round(offset_ql / grid) * grid

        if offset_ql in used_offsets:
            continue
        used_offsets.add(offset_ql)

        pt.insert(offset_ql, n)

    pt.quantize((4,), processOffsets=True, processDurations=True, inPlace=True)
    pt.makeMeasures(inPlace=True)

    sc.insert(0, pt)

    xml_path = out_dir / "drums.musicxml"
    pdf_path = out_dir / "drums.pdf"

    sc.write("musicxml", fp=str(xml_path))
    musicxml_to_pdf(str(xml_path), str(pdf_path))

    return {"midi": None, "musicxml": str(xml_path), "pdf": str(pdf_path)}