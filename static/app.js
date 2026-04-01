const fileInput = document.getElementById("audio-file");
const fileNameText = document.getElementById("file-name");
const uploadForm = document.getElementById("upload-form");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");
const zipLink = document.getElementById("zip-link");
const startBtn = document.getElementById("start-btn");
const uploadBox = document.getElementById("upload-box");
const mixAudio = document.getElementById("mix-audio");

let currentJobId = null;
let pollTimer = null;
let selectedFile = null;
let mixObjectUrl = null;

const STEMS = ["vocals", "drums", "bass", "other"];

const waves = {
  mix: null,
  vocals: null,
  drums: null,
  bass: null,
  other: null,
};

function formatDuration(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return "--:--";
  const total = Math.floor(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function bindAudioDuration(audioId, durationId) {
  const audio = document.getElementById(audioId);
  const durationEl = document.getElementById(durationId);
  if (!audio || !durationEl) return;

  audio.addEventListener("loadedmetadata", () => {
    durationEl.textContent = formatDuration(audio.duration);
  });
}

function destroyWave(key, waveformId) {
  if (waves[key]) {
    waves[key].destroy();
    waves[key] = null;
  }
  const container = document.getElementById(waveformId);
  if (container) {
    container.innerHTML = "";
  }
}

function initWave(key, waveformId, audioId) {
  const container = document.getElementById(waveformId);
  const audioEl = document.getElementById(audioId);

  if (!container || !audioEl || !audioEl.src) return;
  if (typeof WaveSurfer === "undefined") return;

  destroyWave(key, waveformId);

  waves[key] = WaveSurfer.create({
    container: `#${waveformId}`,
    media: audioEl,
    waveColor: "#c7d2fe",
    progressColor: "#4f46e5",
    cursorColor: "#111827",
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    height: 56,
    normalize: true,
  });

  // 注意：这里不要再 waves[key].load(audioEl.src)
  // 因为已经把原生 audio 元素通过 media 传进去了
}

function setSelectedFile(file) {
  selectedFile = file;

  destroyWave("mix", "mix-waveform");

  if (selectedFile) {
    fileNameText.textContent = selectedFile.name;
    startBtn.disabled = false;

    const oldUrl = mixObjectUrl;
    mixObjectUrl = URL.createObjectURL(selectedFile);

    const handleLoaded = () => {
      mixAudio.removeEventListener("loadedmetadata", handleLoaded);
      initWave("mix", "mix-waveform", "mix-audio");

      if (oldUrl) {
        URL.revokeObjectURL(oldUrl);
      }
    };

    mixAudio.addEventListener("loadedmetadata", handleLoaded, { once: true });
    mixAudio.src = mixObjectUrl;
    mixAudio.load();
  } else {
    fileNameText.textContent = "尚未选择文件";
    startBtn.disabled = true;

    if (mixObjectUrl) {
      URL.revokeObjectURL(mixObjectUrl);
      mixObjectUrl = null;
    }

    mixAudio.removeAttribute("src");
    mixAudio.load();
  }
}

function unlockUpload() {
  fileInput.disabled = false;
  fileInput.value = "";   // 关键：清空 input，保证下次选文件一定触发 change
  uploadBox.classList.remove("disabled");
  startBtn.disabled = !selectedFile;
}

function resetUIForNewJob() {
  errorText.textContent = "";
  zipLink.href = "javascript:void(0)";
  zipLink.classList.add("disabled");

  // 不要清掉 mix 波形和 mix 音频
  // 因为用户已经上传成功了，左侧应该还能试听 mix

  STEMS.forEach((stem) => {

    const audio = document.getElementById(`${stem}-audio`);
    const audioDownload = document.getElementById(`${stem}-audio-download`);
    const card = document.getElementById(`${stem}-card`);
    const statusBadge = document.getElementById(`${stem}-status`);
    const durationEl = document.getElementById(`${stem}-duration`);

    if (audio) {
      audio.removeAttribute("src");
      audio.load();
    }

    if (audioDownload) {
      audioDownload.href = "javascript:void(0)";
      audioDownload.classList.add("disabled");
    }

    if (durationEl) {
      durationEl.textContent = "--:--";
    }

    if (card) {
      card.classList.remove("audio-ready", "score-ready");
      card.classList.add("waiting");
    }

    if (statusBadge) {
      statusBadge.textContent = "等待生成";
      statusBadge.classList.remove("hidden", "audio-ready", "score-ready");
      statusBadge.classList.add("waiting");
    }

    if (stem !== "other") {
      const scoreBtn = document.getElementById(`${stem}-score-btn`);
      if (scoreBtn) {
        scoreBtn.href = "javascript:void(0)";
        scoreBtn.classList.add("disabled");
      }
    }
  });
}

function updateUI(job) {
  statusText.textContent = job.message || job.status;
  errorText.textContent = job.error || "";

  STEMS.forEach((stem) => {
    const stemData = job.stems[stem];
    if (!stemData) return;

    const audio = document.getElementById(`${stem}-audio`);
    const audioDownload = document.getElementById(`${stem}-audio-download`);
    const card = document.getElementById(`${stem}-card`);
    const statusBadge = document.getElementById(`${stem}-status`);

    if (stemData.audio_ready && stemData.audio_url) {
      const fullAudioUrl = window.location.origin + stemData.audio_url;

      if (audio && audio.src !== fullAudioUrl) {
        const handleLoaded = () => {
          audio.removeEventListener("loadedmetadata", handleLoaded);
          initWave(stem, `${stem}-waveform`, `${stem}-audio`);
        };

        audio.addEventListener("loadedmetadata", handleLoaded, { once: true });
        audio.src = stemData.audio_url;
        audio.load();
      }

      if (audioDownload) {
        audioDownload.href = stemData.audio_download_url;
        audioDownload.classList.remove("disabled");
      }

      if (card) {
        card.classList.remove("waiting");
        card.classList.add("audio-ready");
      }

      if (statusBadge) {
        statusBadge.textContent = "音频已生成";
        statusBadge.classList.remove("hidden", "waiting", "score-ready");
        statusBadge.classList.add("audio-ready");
      }
    }

    if (stem !== "other") {
      const scoreBtn = document.getElementById(`${stem}-score-btn`);
      if (!scoreBtn) return;

      if (stemData.score_ready && stemData.score_download_url) {
        scoreBtn.href = stemData.score_download_url;
        scoreBtn.classList.remove("disabled");

        if (card) {
          card.classList.add("score-ready");
        }

        if (statusBadge) {
          statusBadge.textContent = "谱面可下载";
          statusBadge.classList.remove("hidden", "waiting", "audio-ready");
          statusBadge.classList.add("score-ready");
        }
      } else {
        scoreBtn.href = "javascript:void(0)";
        scoreBtn.classList.add("disabled");

        if (stemData.audio_ready && statusBadge) {
          statusBadge.textContent = "谱面生成中";
          statusBadge.classList.remove("hidden", "waiting", "score-ready");
          statusBadge.classList.add("audio-ready");
        }
      }
    }
  });

  if (job.zip_url) {
    zipLink.href = job.zip_url;
    zipLink.classList.remove("disabled");
  }

  if (job.status === "done") {
    statusText.textContent = "全部处理完成";
  }
}

async function pollOnce() {
  if (!currentJobId) return;

  try {
    const resp = await fetch(`${STATUS_PREFIX}${currentJobId}`);
    const data = await resp.json();

    if (!resp.ok || !data.ok) {
      if (resp.status === 404 || resp.status === 410) {
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        statusText.textContent = "任务状态已失效，请刷新页面后重新上传。";
        errorText.textContent = data.error || "任务不存在或已过期";
        unlockUpload();
        return;
      }
      throw new Error(data.error || "状态获取失败");
    }

    updateUI(data.job);

    if (data.job.status === "done" || data.job.status === "error") {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      unlockUpload();
    }
  } catch (err) {
    errorText.textContent = err.message || String(err);
    unlockUpload();
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  pollTimer = setInterval(pollOnce, 1500);
}

// 初始化时长监听
bindAudioDuration("mix-audio", "mix-duration");
STEMS.forEach((stem) => {
  bindAudioDuration(`${stem}-audio`, `${stem}-duration`);
});

// 初始状态
startBtn.disabled = true;

// 文件选择
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    setSelectedFile(fileInput.files[0]);
  } else {
    setSelectedFile(null);
  }
});

// 拖拽上传
uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (!fileInput.disabled) {
    uploadBox.classList.add("dragover");
  }
});

uploadBox.addEventListener("dragleave", () => {
  uploadBox.classList.remove("dragover");
});

uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadBox.classList.remove("dragover");

  if (fileInput.disabled) return;

  const files = e.dataTransfer.files;
  if (!files || !files.length) return;

  const file = files[0];
  const isWav = file.name.toLowerCase().endsWith(".wav") || file.type === "audio/wav";
  if (!isWav) {
    errorText.textContent = "仅支持上传 .wav 文件";
    return;
  }

  setSelectedFile(file);
});

// 提交开始分离
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!selectedFile) {
    statusText.textContent = "请先选择 wav 文件。";
    return;
  }

  resetUIForNewJob();

  statusText.textContent = "上传中...";
  errorText.textContent = "";
  startBtn.disabled = true;
  fileInput.disabled = true;
  uploadBox.classList.add("disabled");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch(START_URL, {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "上传失败");
    }

    currentJobId = data.job_id;
    statusText.textContent = "任务已提交，准备开始处理...";
    startPolling();
  } catch (err) {
    statusText.textContent = "启动失败";
    errorText.textContent = err.message || String(err);
    unlockUpload();
  }
});