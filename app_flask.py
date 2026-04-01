from __future__ import annotations

import os
import uuid
import zipfile
import traceback
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort
)
from werkzeug.utils import secure_filename

from src.serving.separator_service import StemSeparator
from src.notation.transcribe import (
    transcribe_vocals,
    transcribe_bass,
    transcribe_drums,
)

# =========================
# Flask app
# =========================
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB

# =========================
# 固定配置
# =========================
OUTPUT_ROOT = Path("/home/ubuntu/CZJ/demo_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CKPT_MAP = {
    "vocals": "/home/ubuntu/CZJ/DTT/checkpoints/vocal.ckpt",
    "drums": "/home/ubuntu/CZJ/DTT/checkpoints/drum.ckpt",
    "bass": "/home/ubuntu/CZJ/DTT/checkpoints/bass.ckpt",
    "other": "/home/ubuntu/CZJ/DTT/checkpoints/other.ckpt",
}

DEVICE = "cuda:0"
SAMPLERATE = 44100
ALLOWED_EXTENSIONS = {"wav"}

# =========================
# 任务状态
# =========================
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_workdir(job_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = OUTPUT_ROOT / f"{ts}_{job_id}"
    (workdir / "uploads").mkdir(parents=True, exist_ok=True)
    (workdir / "stems").mkdir(parents=True, exist_ok=True)
    (workdir / "scores").mkdir(parents=True, exist_ok=True)
    return workdir


def rel_media_url(job_id: str, rel_path: str) -> str:
    return f"/media/{job_id}/{rel_path.replace(os.sep, '/')}"


def rel_download_url(job_id: str, rel_path: str) -> str:
    return f"/download/{job_id}/{rel_path.replace(os.sep, '/')}"


def build_zip(workdir: Path) -> Path:
    zip_path = workdir / "results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in workdir.rglob("*"):
            if p.is_file() and p.name != "results.zip":
                zf.write(p, arcname=p.relative_to(workdir))
    return zip_path


def init_job_state(job_id: str, workdir: Path) -> dict:
    return {
        "job_id": job_id,
        "status": "queued",   # queued / separating / scoring / done / error
        "message": "等待开始",
        "workdir": str(workdir),
        "uploaded_audio": None,
        "zip_url": None,
        "error": None,
        "stems": {
            "vocals": {
                "audio_ready": False,
                "audio_url": None,
                "audio_download_url": None,
                "score_ready": False,
                "score_url": None,
                "score_download_url": None,
                "score_supported": True,
            },
            "drums": {
                "audio_ready": False,
                "audio_url": None,
                "audio_download_url": None,
                "score_ready": False,
                "score_url": None,
                "score_download_url": None,
                "score_supported": True,
            },
            "bass": {
                "audio_ready": False,
                "audio_url": None,
                "audio_download_url": None,
                "score_ready": False,
                "score_url": None,
                "score_download_url": None,
                "score_supported": True,
            },
            "other": {
                "audio_ready": False,
                "audio_url": None,
                "audio_download_url": None,
                "score_ready": False,
                "score_url": None,
                "score_download_url": None,
                "score_supported": False,  # others 不生成谱面
            },
        },
    }


def update_job(job_id: str, updater):
    with JOBS_LOCK:
        updater(JOBS[job_id])


def run_job(job_id: str):
    try:
        with JOBS_LOCK:
            job = JOBS[job_id]
            workdir = Path(job["workdir"])
            uploaded_audio = Path(job["uploaded_audio"])

        update_job(job_id, lambda j: j.update({
            "status": "separating",
            "message": "正在执行四 stem 分离..."
        }))

        separator = StemSeparator(
            ckpt_map=CKPT_MAP,
            device=DEVICE,
            batch_size=1,
            double_chunk=False,
            overlap_add={
                "overlap_rate": 0.5,
                "tmp_root": str((workdir / "tmp").resolve()),
                "samplerate": SAMPLERATE,
            },
            samplerate=SAMPLERATE,
        )

        stem_paths = separator.separate_all(str(uploaded_audio), str(workdir / "stems"))

        # 分离完成后，四个 stem 可播放/下载
        for stem_name, abs_path in stem_paths.items():
            rel_path = str(Path(abs_path).relative_to(workdir))
            update_job(job_id, lambda j, stem_name=stem_name, rel_path=rel_path: j["stems"][stem_name].update({
                "audio_ready": True,
                "audio_url": rel_media_url(job_id, rel_path),
                "audio_download_url": rel_download_url(job_id, rel_path),
            }))

        update_job(job_id, lambda j: j.update({
            "status": "scoring",
            "message": "分离完成，正在自动生成谱面..."
        }))

        scores_dir = workdir / "scores"

        # vocals
        try:
            v = transcribe_vocals(stem_paths["vocals"], str(scores_dir / "vocals"))
            if v.get("pdf"):
                rel_path = str(Path(v["pdf"]).relative_to(workdir))
                update_job(job_id, lambda j: j["stems"]["vocals"].update({
                    "score_ready": True,
                    "score_url": rel_media_url(job_id, rel_path),
                    "score_download_url": rel_download_url(job_id, rel_path),
                }))
        except Exception as e:
            print(f"[WARN] vocals transcription failed: {e}")

        # drums
        try:
            d = transcribe_drums(stem_paths["drums"], str(scores_dir / "drums"))
            if d.get("pdf"):
                rel_path = str(Path(d["pdf"]).relative_to(workdir))
                update_job(job_id, lambda j: j["stems"]["drums"].update({
                    "score_ready": True,
                    "score_url": rel_media_url(job_id, rel_path),
                    "score_download_url": rel_download_url(job_id, rel_path),
                }))
        except Exception as e:
            print(f"[WARN] drums transcription failed: {e}")

        # bass
        try:
            b = transcribe_bass(stem_paths["bass"], str(scores_dir / "bass"))
            if b.get("pdf"):
                rel_path = str(Path(b["pdf"]).relative_to(workdir))
                update_job(job_id, lambda j: j["stems"]["bass"].update({
                    "score_ready": True,
                    "score_url": rel_media_url(job_id, rel_path),
                    "score_download_url": rel_download_url(job_id, rel_path),
                }))
        except Exception as e:
            print(f"[WARN] bass transcription failed: {e}")

        zip_path = build_zip(workdir)
        rel_zip = str(zip_path.relative_to(workdir))
        update_job(job_id, lambda j: j.update({
            "status": "done",
            "message": "全部处理完成。",
            "zip_url": rel_download_url(job_id, rel_zip),
        }))

    except Exception as e:
        traceback.print_exc()
        update_job(job_id, lambda j: j.update({
            "status": "error",
            "message": "处理失败",
            "error": str(e),
        }))

def find_workdir_by_job_id(job_id: str) -> Path | None:
    matches = list(OUTPUT_ROOT.glob(f"*_{job_id}"))
    if not matches:
        return None
    # 取最新一个
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/start")
def api_start():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "没有检测到上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "请选择 wav 文件"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "仅支持 .wav 文件"}), 400

    job_id = uuid.uuid4().hex[:12]
    workdir = make_workdir(job_id)

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix or ".wav"
    saved_path = workdir / "uploads" / f"input{suffix}"
    file.save(saved_path)

    with JOBS_LOCK:
        JOBS[job_id] = init_job_state(job_id, workdir)
        JOBS[job_id]["uploaded_audio"] = str(saved_path)

    t = Thread(target=run_job, args=(job_id,), daemon=True)
    t.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/status/<job_id>")
def api_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        workdir = find_workdir_by_job_id(job_id)
        if workdir:
            return jsonify({
                "ok": False,
                "expired": True,
                "error": "任务状态已过期，请刷新页面并重新上传文件。"
            }), 410
        return jsonify({"ok": False, "error": "任务不存在"}), 404

    return jsonify({"ok": True, "job": job})


@app.get("/media/<job_id>/<path:subpath>")
def media_file(job_id: str, subpath: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job:
        workdir = Path(job["workdir"])
    else:
        workdir = find_workdir_by_job_id(job_id)

    if not workdir:
        abort(404)

    target = workdir / subpath
    if not target.exists():
        abort(404)

    return send_from_directory(workdir, subpath, as_attachment=False)


@app.get("/download/<job_id>/<path:subpath>")
def download_file(job_id: str, subpath: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job:
        workdir = Path(job["workdir"])
    else:
        workdir = find_workdir_by_job_id(job_id)

    if not workdir:
        abort(404)

    target = workdir / subpath
    if not target.exists():
        abort(404)

    return send_from_directory(workdir, subpath, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)