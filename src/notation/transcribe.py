from pathlib import Path
import os
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


def _prepare_basic_pitch_input(wav_path: str, out_dir: Path) -> Path:
    """
    先把输入音频转成 basic_pitch 目标格式：
    - mono
    - sample rate = AUDIO_SAMPLE_RATE
    - wav
    避免 basic_pitch 内部再走 librosa/resampy 重采样链。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_wav = out_dir / "basic_pitch_input.wav"

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


def transcribe_vocals(wav_path: str, out_prefix: str) -> dict:
    out_dir = Path(out_prefix).parent
    os.makedirs(out_dir, exist_ok=True)

    prepared_wav = _prepare_basic_pitch_input(wav_path, out_dir)

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
        return {"midi": None, "musicxml": None}

    midi_path = midi_files[0]
    xml_path = Path(f"{out_prefix}.musicxml")
    midi_to_musicxml(str(midi_path), str(xml_path))
    return {"midi": str(midi_path), "musicxml": str(xml_path)}


def transcribe_bass(wav_path: str, out_prefix: str) -> dict:
    return transcribe_vocals(wav_path, out_prefix)


def transcribe_drums(wav_path: str, out_prefix: str) -> dict:
    # 直接按原始采样率读取，避免 librosa.load(..., sr=22050) 触发重采样崩溃
    y, sr = sf.read(wav_path)

    # 转单声道
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 转成 float32
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

    xml_path = Path(f"{out_prefix}.musicxml")
    sc.write("musicxml", fp=str(xml_path))
    return {"midi": None, "musicxml": str(xml_path)}