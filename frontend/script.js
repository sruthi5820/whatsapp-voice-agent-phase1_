// Frontend logic for the Voice App. Talks to the FastAPI backend.

// When served by FastAPI (http://localhost:8000) use the same origin.
// If index.html is opened straight from disk or another server, fall back to localhost:8000.
// You can also override it by setting window.API_BASE before this script loads.
const API_BASE =
  window.API_BASE ??
  (location.protocol === "file:" || location.port !== "8000" ? "http://localhost:8000" : "");

const $ = (id) => document.getElementById(id);

// ---------- Helpers ----------
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setBusy(button, loadingEl, busy) {
  button.disabled = busy;
  busy ? show(loadingEl) : hide(loadingEl);
}

// Turn any failed response into a readable message.
async function errorMessage(response) {
  try {
    const body = await response.json();
    return [body.error, body.detail].filter(Boolean).join(": ") || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function networkMessage() {
  return `Could not reach the server at ${API_BASE || location.origin}. Is the backend running?`;
}

// Text travels URL-encoded because HTTP headers cannot carry Malayalam directly.
function headerText(response, name) {
  const raw = response.headers.get(name);
  if (!raw) return "";
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    ["stt", "tts", "voice"].forEach((name) => {
      name === tab.dataset.tab ? show($(`tab-${name}`)) : hide($(`tab-${name}`));
    });
  });
});

// ---------- Server status ----------
async function checkHealth() {
  const el = $("serverStatus");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    el.textContent = "● Server online";
    el.className = "status ok";
  } catch {
    el.textContent = "● Server offline";
    el.className = "status down";
  }
}

// ---------- Speech to Text ----------
let mediaRecorder = null;
let microphoneStream = null;
let recordedBlob = null;
let recordingUrl = null;

function clearRecording() {
  recordedBlob = null;
  hide($("recordingPreview"));
  if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  recordingUrl = null;
}

$("sttFile").addEventListener("change", clearRecording);

$("recordBtn").addEventListener("click", async () => {
  hide($("sttError"));
  clearRecording();
  $("sttFile").value = "";

  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    $("sttError").textContent = "Audio recording is not supported in this browser. Try a current version of Chrome, Edge, Firefox, or Safari.";
    show($("sttError"));
    return;
  }

  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    mediaRecorder = mimeType ? new MediaRecorder(microphoneStream, { mimeType }) : new MediaRecorder(microphoneStream);
    const chunks = [];
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      const type = mediaRecorder.mimeType || "audio/webm";
      recordedBlob = new Blob(chunks, { type });
      const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
      recordedBlob.fileName = `recording.${extension}`;
      recordingUrl = URL.createObjectURL(recordedBlob);
      $("recordingPreview").src = recordingUrl;
      show($("recordingPreview"));
      hide($("recordingStatus"));
      show($("recordBtn"));
      hide($("stopRecordBtn"));
      microphoneStream?.getTracks().forEach((track) => track.stop());
      microphoneStream = null;
      $("sttBtn").disabled = false;
    }, { once: true });
    mediaRecorder.start();
    hide($("recordBtn"));
    show($("stopRecordBtn"));
    show($("recordingStatus"));
    $("sttBtn").disabled = true;
  } catch (err) {
    microphoneStream?.getTracks().forEach((track) => track.stop());
    microphoneStream = null;
    $("sttError").textContent = err.name === "NotAllowedError"
      ? "Microphone permission was denied. Allow microphone access and try again."
      : `Could not start recording: ${err.message}`;
    show($("sttError"));
  }
});

$("stopRecordBtn").addEventListener("click", () => {
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
});

$("sttBtn").addEventListener("click", async () => {
  const file = $("sttFile").files[0] || (recordedBlob && new File([recordedBlob], recordedBlob.fileName, { type: recordedBlob.type }));
  hide($("sttError"));
  hide($("sttResult"));

  if (!file) {
    $("sttError").textContent = "Choose an audio file or record your voice first.";
    show($("sttError"));
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("model", $("sttModel").value);

  setBusy($("sttBtn"), $("sttLoading"), true);
  try {
    const res = await fetch(`${API_BASE}/stt`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorMessage(res));
    const data = await res.json();
    $("sttText").value = data.text || "(no speech recognised)";
    console.log(`[STT] ${data.model} (${data.language}) transcript:`, JSON.stringify(data.text || "(empty)"));
    $("sttMeta").textContent = `Model: ${data.model} · ${data.processing_seconds}s`;
    show($("sttResult"));
  } catch (err) {
    $("sttError").textContent = err instanceof TypeError ? networkMessage() : err.message;
    show($("sttError"));
  } finally {
    setBusy($("sttBtn"), $("sttLoading"), false);
  }
});

// ---------- Text to Speech ----------
let lastAudioUrl = null;

$("ttsBtn").addEventListener("click", async () => {
  const text = $("ttsText").value.trim();
  hide($("ttsError"));
  hide($("ttsResult"));

  if (!text) {
    $("ttsError").textContent = "Please enter some Malayalam text.";
    show($("ttsError"));
    return;
  }

  setBusy($("ttsBtn"), $("ttsLoading"), true);
  try {
    const res = await fetch(`${API_BASE}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(await errorMessage(res));

    const blob = await res.blob();
    if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl); // free the previous audio
    lastAudioUrl = URL.createObjectURL(blob);

    $("ttsAudio").src = lastAudioUrl;
    $("ttsDownload").href = lastAudioUrl;
    show($("ttsResult"));
    $("ttsAudio").play().catch(() => {}); // autoplay may be blocked; the controls still work
  } catch (err) {
    $("ttsError").textContent = err instanceof TypeError ? networkMessage() : err.message;
    show($("ttsError"));
  } finally {
    setBusy($("ttsBtn"), $("ttsLoading"), false);
  }
});

// ---------- Voice Assistant (STT -> Ollama -> TTS) ----------
let voiceMediaRecorder = null;
let voiceMicStream = null;
let voiceRecordedBlob = null;
let voiceRecordingUrl = null;

function clearVoiceRecording() {
  voiceRecordedBlob = null;
  hide($("voiceRecordingPreview"));
  if (voiceRecordingUrl) URL.revokeObjectURL(voiceRecordingUrl);
  voiceRecordingUrl = null;
}

$("voiceFile").addEventListener("change", clearVoiceRecording);

$("voiceRecordBtn").addEventListener("click", async () => {
  hide($("voiceError"));
  clearVoiceRecording();
  $("voiceFile").value = "";

  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    $("voiceError").textContent = "Audio recording is not supported in this browser. Try Chrome, Edge, Firefox, or Safari.";
    show($("voiceError"));
    return;
  }

  try {
    voiceMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    voiceMediaRecorder = mimeType ? new MediaRecorder(voiceMicStream, { mimeType }) : new MediaRecorder(voiceMicStream);
    const chunks = [];
    voiceMediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    voiceMediaRecorder.addEventListener("stop", () => {
      const type = voiceMediaRecorder.mimeType || "audio/webm";
      voiceRecordedBlob = new Blob(chunks, { type });
      const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
      voiceRecordedBlob.fileName = `recording.${extension}`;
      voiceRecordingUrl = URL.createObjectURL(voiceRecordedBlob);
      $("voiceRecordingPreview").src = voiceRecordingUrl;
      show($("voiceRecordingPreview"));
      hide($("voiceRecordingStatus"));
      show($("voiceRecordBtn"));
      hide($("voiceStopRecordBtn"));
      voiceMicStream?.getTracks().forEach((track) => track.stop());
      voiceMicStream = null;
      $("voiceBtn").disabled = false;
    }, { once: true });
    voiceMediaRecorder.start();
    hide($("voiceRecordBtn"));
    show($("voiceStopRecordBtn"));
    show($("voiceRecordingStatus"));
    $("voiceBtn").disabled = true;
  } catch (err) {
    voiceMicStream?.getTracks().forEach((track) => track.stop());
    voiceMicStream = null;
    $("voiceError").textContent = err.name === "NotAllowedError"
      ? "Microphone permission was denied. Allow microphone access and try again."
      : `Could not start recording: ${err.message}`;
    show($("voiceError"));
  }
});

$("voiceStopRecordBtn").addEventListener("click", () => {
  if (voiceMediaRecorder?.state === "recording") voiceMediaRecorder.stop();
});

$("voiceBtn").addEventListener("click", async () => {
  const file = $("voiceFile").files[0] || (voiceRecordedBlob && new File([voiceRecordedBlob], voiceRecordedBlob.fileName, { type: voiceRecordedBlob.type }));
  hide($("voiceError"));
  hide($("voiceResult"));

  if (!file) {
    $("voiceError").textContent = "Choose an audio file or record your voice first.";
    show($("voiceError"));
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("stt_model", $("voiceSttModel").value);
  form.append("llm_model", $("voiceLlmModel").value);
  form.append("tts_model", $("voiceTtsModel").value);
  const system = $("voiceSystem").value.trim();
  if (system) form.append("system", system);

  setBusy($("voiceBtn"), $("voiceLoading"), true);
  try {
    const res = await fetch(`${API_BASE}/voice`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorMessage(res));

    const blob = await res.blob();
    if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
    lastAudioUrl = URL.createObjectURL(blob);

    $("voiceText").value = headerText(res, "X-Stt-Text") || "";
    $("voiceReply").value = headerText(res, "X-Llm-Reply") || "";
    $("voiceAudio").src = lastAudioUrl;
    $("voiceDownload").href = lastAudioUrl;
    show($("voiceResult"));
    $("voiceAudio").play().catch(() => {});
  } catch (err) {
    $("voiceError").textContent = err instanceof TypeError ? networkMessage() : err.message;
    show($("voiceError"));
  } finally {
    setBusy($("voiceBtn"), $("voiceLoading"), false);
  }
});

// ---------- Phone Call: AI greets first, then continuous listen → reply ----------
const CALL_GREETING = window.CALL_GREETING ?? "ഹലോ! നിങ്ങൾ ആരാണ്?";
const CALL_SILENCE_MS = 1200;      // stop a turn after this much silence
const CALL_MAX_TURN_MS = 15000;    // hard cap on one caller turn
const CALL_MIN_SPEECH_MS = 300;    // ignore blips shorter than this
const CALL_NO_SPEECH_MS = 12000;   // reset the turn if the mic never crosses the threshold
const CALL_ENERGY_THRESHOLD = 0.02;
const CALL_SYSTEM =
  "You are a helpful voice assistant on a phone call. Reply in Malayalam with " +
  "one short sentence of about 15-20 words. English technical words are fine; " +
  "natural Malayalam-English mixing is allowed. Keep it natural for " +
  "text-to-speech. No markdown, no bullet points, no special characters.";

let callActive = false;
let callMicStream = null;
let callAudioCtx = null;
let callAnalyser = null;
let callAnalyserSink = null;
let callAnalyserData = null;
let callRecorder = null;
let callChunks = [];
let callDiscardTurn = false;
let callVadFrame = null;
let callHeardSpeech = false;
let callSpeechStart = 0;
let callSilenceStart = 0;
let callAudioUrl = null;
let callAudioDone = null;
let callSession = 0;
let callHistory = [];
let callSuspended = false;
let callBusy = false;
let callLastLevelLog = 0;
let callListenStart = 0;
let callPeakLevel = 0;
let callZeroLogs = 0;
let callDeviceId = "";

function setCallStatus(text, cls = "") {
  $("callStatus").textContent = text;
  $("callStatus").className = `status ${cls}`.trim();
}

function callMimeType() {
  return ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
    .find((type) => MediaRecorder.isTypeSupported(type));
}

function recorderOptions() {
  const mimeType = callMimeType();
  return mimeType ? { mimeType } : undefined;
}

async function populateMicDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const select = $("callMicSelect");
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter((device) => device.kind === "audioinput");
    console.log(
      "[Call] audio input devices:",
      mics.map((device, index) => ({
        index,
        id: device.deviceId,
        label: device.label || "(label hidden until mic permission is granted)",
      }))
    );
    if (!select) return;

    const current = select.value || callDeviceId;
    select.innerHTML = "";
    const def = document.createElement("option");
    def.value = "";
    def.textContent = "System default";
    select.appendChild(def);
    mics.forEach((device, index) => {
      const opt = document.createElement("option");
      opt.value = device.deviceId;
      opt.textContent = device.label || `Microphone ${index + 1}`;
      select.appendChild(opt);
    });
    if (current && mics.some((device) => device.deviceId === current)) {
      select.value = current;
    }
  } catch (err) {
    console.warn("[Call] could not list microphones:", err);
  }
}

function playCallAudio(url) {
  return new Promise((resolve) => {
    const audio = $("callAudio");
    const done = () => {
      audio.removeEventListener("ended", done);
      audio.removeEventListener("error", done);
      if (callAudioDone === done) callAudioDone = null;
      resolve();
    };
    callAudioDone = done;
    audio.addEventListener("ended", done);
    audio.addEventListener("error", done);
    audio.src = url;
    audio.play().catch(() => {
      // Autoplay can be blocked; the controls remain usable.
      setCallStatus("Tap play on the audio player to hear the AI", "down");
      done();
    });
  });
}

async function synthesizeCallText(text) {
  const res = await fetch(`${API_BASE}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return URL.createObjectURL(await res.blob());
}

function stopVad() {
  if (callVadFrame) cancelAnimationFrame(callVadFrame);
  callVadFrame = null;
}

function startVad() {
  if (!callAnalyser) {
    console.warn("[Call] VAD not started: no microphone analyser");
    return;
  }
  const data = callAnalyserData;
  const tick = () => {
    if (!callActive || !callRecorder || callRecorder.state !== "recording") return;

    callAnalyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length);
    const now = performance.now();
    const speaking = rms >= CALL_ENERGY_THRESHOLD;
    callPeakLevel = Math.max(callPeakLevel, rms);

    // Periodic mic-level readout so you can confirm the browser is capturing
    // audio: if rms stays ~0.0000 while you talk, the mic is not working.
    if (now - callLastLevelLog >= 500) {
      callLastLevelLog = now;
      console.log(
        `[Call] mic level rms=${rms.toFixed(4)} peak=${callPeakLevel.toFixed(4)} ` +
        `(threshold=${CALL_ENERGY_THRESHOLD}) ${speaking ? "speech" : "silence"}`
      );
      if (rms < 0.00001) {
        callZeroLogs += 1;
        if (callZeroLogs === 4) {
          console.error(
            "[Call] the microphone has delivered pure digital silence for 2s. " +
            "This is not a VAD bug: check that the right input device is selected, " +
            "that it is not muted in the OS, and that no other app is holding it."
          );
        }
      } else {
        callZeroLogs = 0;
      }
    }

    if (speaking) {
      if (!callHeardSpeech) {
        callHeardSpeech = true;
        callSpeechStart = now;
        console.log("[Call] voice detected — recording this turn");
      }
      callSilenceStart = 0;
    } else if (callHeardSpeech) {
      if (!callSilenceStart) {
        callSilenceStart = now;
        console.log("[Call] speech ended — waiting for silence timeout");
      }
      if (now - callSilenceStart >= CALL_SILENCE_MS) {
        console.log("[Call] silence timeout reached — sending turn");
        if (callRecorder?.state === "recording") callRecorder.stop();
        return;
      }
    }

    const spokeLongEnough = callHeardSpeech && now - callSpeechStart >= CALL_MIN_SPEECH_MS;
    if (spokeLongEnough && now - callSpeechStart >= CALL_MAX_TURN_MS) {
      console.warn("[Call] max turn length reached — sending turn");
      if (callRecorder?.state === "recording") callRecorder.stop();
      return;
    }

    // Never stay in "Listening…" forever if the mic level never crosses the
    // threshold (wrong input device, muted mic, or threshold too high).
    if (!callHeardSpeech && callListenStart && now - callListenStart >= CALL_NO_SPEECH_MS) {
      console.warn(
        `[Call] no speech above threshold for ${CALL_NO_SPEECH_MS / 1000}s ` +
        `(peak=${callPeakLevel.toFixed(4)}); resetting the listening turn`
      );
      if (callRecorder?.state === "recording") callRecorder.stop();
      return;
    }

    callVadFrame = requestAnimationFrame(tick);
  };
  callVadFrame = requestAnimationFrame(tick);
}

function startListening() {
  if (!callActive || !callMicStream || callSuspended) {
    console.log("[Call] startListening skipped", {
      callActive,
      hasMic: Boolean(callMicStream),
      suspended: callSuspended,
    });
    return;
  }
  if (callRecorder?.state === "recording") return;

  hide($("callError"));
  setCallStatus("● Listening… speak now", "ok");
  callDiscardTurn = false;
  callChunks = [];
  callHeardSpeech = false;
  callSpeechStart = 0;
  callSilenceStart = 0;
  callLastLevelLog = 0;
  callListenStart = performance.now();
  callPeakLevel = 0;
  callZeroLogs = 0;
  const session = callSession;

  const recorder = new MediaRecorder(callMicStream, recorderOptions());
  callRecorder = recorder;
  console.log("[Call] listening started", {
    mimeType: recorder.mimeType,
    streamActive: callMicStream.active,
    trackState: callMicStream.getAudioTracks()[0]?.readyState,
  });
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) {
      callChunks.push(event.data);
      const total = callChunks.reduce((n, chunk) => n + chunk.size, 0);
      console.log("[Call] audio chunk captured", event.data.size, "bytes; total", total);
    }
  });
  recorder.addEventListener("stop", async () => {
    stopVad();
    if (callRecorder === recorder) callRecorder = null;
    if (session !== callSession) return;

    const shouldSend = callActive && !callDiscardTurn && callHeardSpeech;
    const type = recorder.mimeType || "audio/webm";
    const blob = new Blob(callChunks, { type });
    const totalBytes = blob.size;
    callChunks = [];
    console.log("[Call] recording stopped", {
      heardSpeech: callHeardSpeech,
      bytes: totalBytes,
      willSend: shouldSend,
    });

    // Decode the recorded blob to verify whether it actually contains audio.
    // This separates a VAD/analyser bug from "the microphone is sending silence".
    try {
      const decoded = await callAudioCtx.decodeAudioData(await blob.arrayBuffer());
      const samples = decoded.getChannelData(0);
      let sum = 0;
      let peak = 0;
      for (let i = 0; i < samples.length; i += 1) {
        sum += samples[i] * samples[i];
        peak = Math.max(peak, Math.abs(samples[i]));
      }
      const rms = Math.sqrt(sum / Math.max(1, samples.length));
      console.log("[Call] recorded audio check", {
        seconds: Number(decoded.duration.toFixed(2)),
        sampleRate: decoded.sampleRate,
        rms: Number(rms.toFixed(5)),
        peak: Number(peak.toFixed(5)),
        verdict: peak < 0.001
          ? "SILENT — microphone/device problem, not the VAD"
          : "audio present",
      });
    } catch (err) {
      console.warn("[Call] could not decode recorded audio for verification:", err);
    }

    callDiscardTurn = false;

    if (!shouldSend) {
      console.warn("[Call] nothing sent: no speech detected in this turn");
      if (callActive && !callSuspended) startListening();
      return;
    }

    const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
    await sendCallTurn(blob, `call.${extension}`);
  }, { once: true });

  try {
    recorder.start();
  } catch (err) {
    console.error("[Call] MediaRecorder could not start:", err);
    if (callRecorder === recorder) callRecorder = null;
    setCallStatus("● Microphone could not start — check permissions", "down");
    return;
  }
  startVad();
}

function callSystemPrompt() {
  if (!callHistory.length) return CALL_SYSTEM;
  const transcript = callHistory
    .slice(-6)
    .map((turn) => `Caller: ${turn.user}\nAssistant: ${turn.ai}`)
    .join("\n");
  return `${CALL_SYSTEM}\n\nConversation so far:\n${transcript}`;
}

async function sendCallTurn(blob, filename) {
  if (!callActive || callBusy) return;
  callBusy = true;
  const session = callSession;

  const form = new FormData();
  form.append("file", new File([blob], filename, { type: blob.type }));
  if (callHistory.length) form.append("system", callSystemPrompt());
  console.log("[Call] sending turn to /voice", {
    filename,
    bytes: blob.size,
    type: blob.type,
  });

  setCallStatus("● AI is thinking…");
  try {
    const res = await fetch(`${API_BASE}/voice`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorMessage(res));
    if (!callActive || session !== callSession) return;

    const audioBlob = await res.blob();
    if (!callActive || session !== callSession) return;
    if (callAudioUrl) URL.revokeObjectURL(callAudioUrl);
    callAudioUrl = URL.createObjectURL(audioBlob);

    const sttText = headerText(res, "X-Stt-Text");
    const llmReply = headerText(res, "X-Llm-Reply");
    $("callText").value = sttText || "(no speech recognised)";
    $("callReply").value = llmReply || "";
    console.log("[Call] STT transcript (Malayalam):", JSON.stringify(sttText || "(empty)"));
    console.log("[Call] LLM reply:", JSON.stringify(llmReply));
    if (!sttText) {
      console.warn("[Call] server returned an empty transcription for this turn");
    }
    $("callDownload").href = callAudioUrl;
    show($("callResult"));

    if ($("callText").value && $("callReply").value) {
      callHistory.push({ user: $("callText").value, ai: $("callReply").value });
    }

    setCallStatus("● AI is speaking…", "ok");
    await playCallAudio(callAudioUrl);
    if (callActive && session === callSession) startListening();
  } catch (err) {
    if (!callActive || session !== callSession) return;
    console.error("[Call] turn failed:", err);
    $("callError").textContent = err instanceof TypeError ? networkMessage() : err.message;
    show($("callError"));
    setCallStatus("● Could not respond — listening again…", "down");
    setTimeout(() => { if (callActive && session === callSession) startListening(); }, 2000);
  } finally {
    callBusy = false;
  }
}

async function sendTypedTurn() {
  const text = $("callText").value.trim();
  if (!text || callBusy) return;
  if (!callActive) {
    setCallStatus("● Start the call first", "down");
    return;
  }

  callBusy = true;
  $("callSendBtn").disabled = true;
  const session = callSession;
  const wasActive = callActive;
  if (wasActive) {
    callSuspended = true;
    stopVad();
    if (callRecorder?.state === "recording") {
      callDiscardTurn = true;
      try { callRecorder.stop(); } catch { /* already stopped */ }
    }
  }

  const form = new FormData();
  form.append("text", text);
  if (callHistory.length) form.append("system", callSystemPrompt());

  hide($("callError"));
  setCallStatus("● AI is thinking…");
  let ok = false;
  try {
    const res = await fetch(`${API_BASE}/voice/text`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await errorMessage(res));
    if (session !== callSession) return;

    const audioBlob = await res.blob();
    if (session !== callSession) return;
    if (callAudioUrl) URL.revokeObjectURL(callAudioUrl);
    callAudioUrl = URL.createObjectURL(audioBlob);

    $("callText").value = headerText(res, "X-Input-Text") || text;
    $("callReply").value = headerText(res, "X-Llm-Reply") || "";
    $("callDownload").href = callAudioUrl;
    show($("callResult"));

    if ($("callReply").value) {
      callHistory.push({ user: $("callText").value, ai: $("callReply").value });
    }
    ok = true;

    setCallStatus("● AI is speaking…", "ok");
    await playCallAudio(callAudioUrl);
  } catch (err) {
    if (session !== callSession) return;
    $("callError").textContent = err instanceof TypeError ? networkMessage() : err.message;
    show($("callError"));
    setCallStatus("● Could not respond", "down");
  } finally {
    callBusy = false;
    $("callSendBtn").disabled = false;
    if (session === callSession && callActive) {
      // Stay in typing mode after a typed turn so the microphone cannot
      // compete with TTS or transcribe background noise.
      callSuspended = true;
      if (ok) setCallStatus("● Typed reply ready — type again or click 🎙 Speak");
    } else if (ok) {
      setCallStatus("Typed reply ready");
    }
  }
}

function pauseCallListening() {
  if (!callActive) return;
  callSuspended = true;
  stopVad();
  if (callRecorder?.state === "recording") {
    callDiscardTurn = true;
    try { callRecorder.stop(); } catch { /* already stopped */ }
  }
  setCallStatus("● Typing… press Send when ready");
}

function resumeCallListening() {
  if (!callActive) {
    setCallStatus("● Start the call first", "down");
    return;
  }
  if (!callMicStream) {
    setCallStatus("● Microphone unavailable — use the typed box", "down");
    return;
  }
  callSuspended = false;
  startListening();
}

async function beginCall() {
  hide($("callError"));
  hide($("callResult"));
  $("callText").value = "";
  $("callReply").value = "";
  $("callAudio").removeAttribute("src");

  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    $("callError").textContent = "Audio recording is not supported in this browser. Use Chrome, Edge, Firefox, or Safari.";
    show($("callError"));
    return;
  }

  setCallStatus("● Connecting…");
  try {
    callSession += 1;
    callDiscardTurn = false;
    callSuspended = false;
    callHistory = [];
    const selectedMic = $("callMicSelect")?.value || callDeviceId;
    const audioConstraints = {
      // Browser DSP makes VAD and STT far more reliable on noisy calls.
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };
    if (selectedMic) audioConstraints.deviceId = { exact: selectedMic };

    try {
      callMicStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    } catch (err) {
      if (!selectedMic || err.name !== "OverconstrainedError") throw err;
      console.warn("[Call] selected microphone unavailable; falling back to the system default", err);
      delete audioConstraints.deviceId;
      callMicStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    }
    $("callMicSelect").disabled = true;
    await populateMicDevices();
    console.log("[Call] microphone stream acquired", {
      active: callMicStream.active,
      tracks: callMicStream.getAudioTracks().map((track) => ({
        label: track.label,
        enabled: track.enabled,
        muted: track.muted,
        readyState: track.readyState,
        settings: track.getSettings(),
      })),
    });
    callMicStream.getAudioTracks().forEach((track) => {
      track.addEventListener("mute", () => console.warn("[Call] mic track muted by the browser/OS"));
      track.addEventListener("unmute", () => console.log("[Call] mic track unmuted"));
      track.addEventListener("ended", () => console.warn("[Call] mic track ended"));
    });

    callAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (callAudioCtx.state === "suspended") await callAudioCtx.resume();
    console.log("[Call] audio context", {
      state: callAudioCtx.state,
      sampleRate: callAudioCtx.sampleRate,
    });
    const source = callAudioCtx.createMediaStreamSource(callMicStream);
    callAnalyser = callAudioCtx.createAnalyser();
    callAnalyser.fftSize = 1024;
    callAnalyser.smoothingTimeConstant = 0.8;
    source.connect(callAnalyser);
    // An AnalyserNode that is not connected to the destination may never be
    // pulled by the audio graph (browser-dependent), which made getFloatTime-
    // DomainData return all zeros and the VAD listen forever. Route it through
    // a muted gain node so it is processed without echoing the mic to speakers.
    callAnalyserSink = callAudioCtx.createGain();
    callAnalyserSink.gain.value = 0;
    callAnalyser.connect(callAnalyserSink);
    callAnalyserSink.connect(callAudioCtx.destination);
    callAnalyserData = new Float32Array(callAnalyser.fftSize);

    callActive = true;
    console.log("[Call] call started; the mic is live and VAD will log levels every 500 ms");
    $("callBtn").textContent = "⏹ End Call";
    $("callBtn").classList.add("call-stop");

    // The AI speaks first, then the caller's turn is recorded automatically.
    setCallStatus("● AI is greeting you…");
    const greetingUrl = await synthesizeCallText(CALL_GREETING);
    if (!callActive) {
      URL.revokeObjectURL(greetingUrl);
      return;
    }

    $("callReply").value = CALL_GREETING;
    show($("callResult"));
    await playCallAudio(greetingUrl);
    URL.revokeObjectURL(greetingUrl);
    callHistory = [{ user: "(call started)", ai: CALL_GREETING }];
    if (callActive) startListening();
  } catch (err) {
    console.error("[Call] could not start the call:", err);
    callMicStream?.getTracks().forEach((track) => track.stop());
    callMicStream = null;
    callActive = false;
    $("callMicSelect").disabled = false;
    $("callBtn").textContent = "▶ Start Call";
    $("callBtn").classList.remove("call-stop");
    $("callError").textContent = err.name === "NotAllowedError"
      ? "Microphone permission was denied. Allow microphone access and try again."
      : err instanceof TypeError ? networkMessage() : `Could not start the call: ${err.message}`;
    show($("callError"));
    setCallStatus("");
  }
}

function endCall() {
  console.log("[Call] ending call");
  callActive = false;
  callSuspended = false;
  callSession += 1;
  callDiscardTurn = true;
  stopVad();
  if (callAudioDone) {
    const done = callAudioDone;
    callAudioDone = null;
    done();
  }
  if (callRecorder?.state === "recording") {
    try { callRecorder.stop(); } catch { /* already stopped */ }
  }
  callRecorder = null;
  callMicStream?.getTracks().forEach((track) => track.stop());
  callMicStream = null;
  $("callMicSelect").disabled = false;
  if (callAudioCtx) {
    callAudioCtx.close().catch(() => {});
    callAudioCtx = null;
  }
  try { callAnalyserSink?.disconnect(); } catch { /* ignore */ }
  callAnalyserSink = null;
  callAnalyser = null;
  callAnalyserData = null;
  if (callAudioUrl) URL.revokeObjectURL(callAudioUrl);
  callAudioUrl = null;
  const audio = $("callAudio");
  audio.pause();
  audio.removeAttribute("src");

  $("callBtn").textContent = "▶ Start Call";
  $("callBtn").classList.remove("call-stop");
  setCallStatus("Call ended");
}

$("callBtn").addEventListener("click", () => {
  if (callActive) endCall();
  else beginCall();
});

$("callMicSelect").addEventListener("change", () => {
  callDeviceId = $("callMicSelect").value;
  console.log("[Call] selected microphone:", callDeviceId || "system default");
});

$("callSendBtn").addEventListener("click", sendTypedTurn);
$("callListenBtn").addEventListener("click", resumeCallListening);
$("callText").addEventListener("focus", pauseCallListening);
$("callText").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    sendTypedTurn();
  }
});

populateMicDevices();
navigator.mediaDevices?.addEventListener?.("devicechange", populateMicDevices);
checkHealth();
