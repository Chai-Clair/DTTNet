from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
import librosa
import soundfile as sf

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
        "未找到 MuseScore 命令，请先安装 musescore/mscore3，"
        "或设置环境变量 MUSESCORE_BIN"
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

# quarterLength 候选：16分、8分三连音、8分、4分三连音2单位、4分、附点4分、2分、全音符
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
    frame_times = librosa.frames_to_time(
        np.arange(len(midi_track)),
        sr=sr,
        hop_length=VOCAL_HOP_LENGTH
    )

    segments = []
    start_i = None
    midi_buf = []
    last_valid_i = None
    unvoiced_gap = 0

    def flush_segment(end_i: int):
        nonlocal start_i, midi_buf, last_valid_i, unvoiced_gap
        if start_i is None or last_valid_i is None or len(midi_buf) < 3:
            start_i = None
            midi_buf = []
            last_valid_i = None
            unvoiced_gap = 0
            return

        start_t = float(frame_times[start_i])
        end_t = float(frame_times[last_valid_i] + VOCAL_HOP_LENGTH / sr)
        if end_t - start_t < 0.08:
            start_i = None
            midi_buf = []
            last_valid_i = None
            unvoiced_gap = 0
            return

        pitch_midi = float(np.median(midi_buf))
        segments.append({
            "start_sec": start_t,
            "end_sec": end_t,
            "midi": pitch_midi,
        })

        start_i = None
        midi_buf = []
        last_valid_i = None
        unvoiced_gap = 0

    for i in range(len(midi_track)):
        active = (
            bool(voiced_flag[i])
            and np.isfinite(midi_track[i])
            and np.isfinite(voiced_prob[i])
            and voiced_prob[i] >= 0.55
        )

        if start_i is None:
            if active:
                start_i = i
                midi_buf = [float(midi_track[i])]
                last_valid_i = i
                unvoiced_gap = 0
            continue

        if active:
            cur_midi = float(midi_track[i])
            recent_med = float(np.median(midi_buf[-8:])) if midi_buf else cur_midi

            should_split = False
            if i in onset_set and (i - start_i) >= 6:
                should_split = True
            elif abs(cur_midi - recent_med) >= 1.2 and (i - start_i) >= 6:
                should_split = True

            if should_split:
                flush_segment(i - 1)
                start_i = i
                midi_buf = [cur_midi]
                last_valid_i = i
                unvoiced_gap = 0
            else:
                midi_buf.append(cur_midi)
                last_valid_i = i
                unvoiced_gap = 0
        else:
            unvoiced_gap += 1
            if unvoiced_gap >= 3:
                flush_segment(i - unvoiced_gap)
            # 否则先容忍一个很短的无声缝隙

    if start_i is not None:
        flush_segment(len(midi_track) - 1)

    return segments


def _merge_same_pitch_neighbors(segments):
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        same_pitch = abs(seg["midi"] - prev["midi"]) < 0.5
        small_gap = (seg["start_sec"] - prev["end_sec"]) < 0.10

        if same_pitch and small_gap:
            prev["end_sec"] = seg["end_sec"]
        else:
            merged.append(seg)

    return merged

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

    for seg in segments:
        start_sec = max(0.0, seg["start_sec"] - anchor_sec)
        dur_sec = max(0.06, seg["end_sec"] - seg["start_sec"])

        offset_ql = _quantize_offset_ql(start_sec, beat_sec)
        dur_ql = _quantize_duration_ql(dur_sec, beat_sec)

        n = note.Note()
        n.pitch.midi = int(round(seg["midi"]))
        n.quarterLength = dur_ql

        pt.insert(offset_ql, n)

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