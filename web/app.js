const elements = {
  fileInput: document.querySelector("#audio-file"),
  dropZone: document.querySelector("#drop-zone"),
  recordButton: document.querySelector("#record-button"),
  recordTime: document.querySelector("#record-time"),
  recordHint: document.querySelector("#record-hint"),
  previewWrap: document.querySelector("#audio-preview-wrap"),
  preview: document.querySelector("#audio-preview"),
  fileName: document.querySelector("#file-name"),
  fileSize: document.querySelector("#file-size"),
  removeAudio: document.querySelector("#remove-audio"),
  analyzeButton: document.querySelector("#analyze-button"),
  streamingPanel: document.querySelector("#streaming-panel"),
  streamingStatus: document.querySelector("#streaming-status"),
  streamingText: document.querySelector("#streaming-text"),
  streamingTranslationLanguage: document.querySelector("#streaming-translation-language"),
  streamingTranslationEmpty: document.querySelector("#streaming-translation-empty"),
  streamingTranslations: document.querySelector("#streaming-translations"),
  streamingEmotionEmpty: document.querySelector("#streaming-emotion-empty"),
  streamingEmotions: document.querySelector("#streaming-emotions"),
  conversationMode: document.querySelector("#conversation-mode"),
  ttsVoice: document.querySelector("#tts-voice"),
  conversationHint: document.querySelector("#conversation-hint"),
  streamingConversationPanel: document.querySelector("#streaming-conversation-panel"),
  streamingConversationEmpty: document.querySelector("#streaming-conversation-empty"),
  streamingConversationLog: document.querySelector("#streaming-conversation-log"),
  targetLanguage: document.querySelector("#target-language"),
  configurationNotice: document.querySelector("#configuration-notice"),
  emptyState: document.querySelector("#empty-state"),
  loadingState: document.querySelector("#loading-state"),
  loadingTitle: document.querySelector("#loading-title"),
  loadingDescription: document.querySelector("#loading-description"),
  results: document.querySelector("#results"),
  analysisErrors: document.querySelector("#analysis-errors"),
  agentActivity: document.querySelector("#agent-activity"),
  agentReviewNotes: document.querySelector("#agent-review-notes"),
  newAnalysis: document.querySelector("#new-analysis"),
  recordTimer: null,
};

let selectedFile = null;
let audioObjectUrl = null;
let microphoneStream = null;
let microphoneSource = null;
let audioContext = null;
let audioProcessor = null;
let streamingSocket = null;
let streamingActive = false;
let streamingFlushResolve = null;
let streamingReadyResolve = null;
let streamingReadyReject = null;
let streamingConfigResolve = null;
let streamingConfigReject = null;
let recordingStartedAt = 0;
let audioDuration = null;
let currentTranscript = "";
let loadingTimer = null;
let localTtsVoices = [];
let assistantSpeechQueue = [];
let assistantSpeechActive = false;
let assistantSpeechGeneration = 0;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function showNotice(message, className = "notice-info") {
  elements.configurationNotice.className = `notice ${className}`;
  elements.configurationNotice.textContent = message;
  elements.configurationNotice.hidden = false;
}

function setSelectedFile(file) {
  if (!file) return;
  if (streamingActive) void stopStreamingRecording();
  if (!file.type.startsWith("audio/") && !/\.(wav|mp3|mpeg|mpga|m4a|ogg|opus|webm|flac)$/i.test(file.name)) {
    showNotice("That file does not look like an audio recording. Choose a WAV, MP3, M4A, OGG, or similar audio file.", "notice-error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showNotice("This recording is over 25 MB. Choose a smaller audio file.", "notice-error");
    return;
  }
  selectedFile = file;
  audioDuration = null;
  currentTranscript = "";
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatBytes(file.size);
  elements.previewWrap.hidden = false;
  elements.analyzeButton.disabled = false;
  elements.streamingPanel.hidden = true;
  elements.results.hidden = true;
  elements.loadingState.hidden = true;
  elements.emptyState.hidden = false;
  elements.configurationNotice.hidden = true;
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = URL.createObjectURL(file);
  elements.preview.src = audioObjectUrl;
  elements.preview.onloadedmetadata = () => {
    if (Number.isFinite(elements.preview.duration)) audioDuration = elements.preview.duration;
  };
}

function clearSelectedFile() {
  selectedFile = null;
  audioDuration = null;
  currentTranscript = "";
  elements.fileInput.value = "";
  elements.preview.removeAttribute("src");
  elements.preview.load();
  elements.previewWrap.hidden = true;
  elements.analyzeButton.disabled = true;
  elements.streamingPanel.hidden = true;
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = null;
  elements.results.hidden = true;
  elements.loadingState.hidden = true;
  elements.emptyState.hidden = false;
  elements.configurationNotice.hidden = true;
}

function setRecordingUi(isRecording) {
  elements.recordButton.classList.toggle("recording", isRecording);
  elements.recordButton.setAttribute("aria-label", isRecording ? "Stop live transcription" : "Stream from microphone");
  elements.recordButton.querySelector("span:nth-child(2)").textContent = isRecording ? "Stop live transcription" : "Stream from microphone";
  elements.recordTime.textContent = isRecording ? "00:00" : "";
}

function resampleAudio(samples, sourceRate, targetRate = 16000) {
  if (sourceRate === targetRate) return samples;
  const ratio = sourceRate / targetRate;
  const output = new Float32Array(Math.floor(samples.length / ratio));
  for (let index = 0; index < output.length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(samples.length, Math.floor((index + 1) * ratio));
    if (end <= start) {
      output[index] = samples[Math.min(start, samples.length - 1)];
      continue;
    }
    let total = 0;
    for (let sample = start; sample < end; sample += 1) total += samples[sample];
    output[index] = total / Math.max(1, end - start);
  }
  return output;
}

function stopAudioCapture() {
  window.clearInterval(elements.recordTimer);
  if (audioProcessor) {
    audioProcessor.port.onmessage = null;
    audioProcessor.disconnect();
    audioProcessor = null;
  }
  microphoneSource?.disconnect();
  microphoneSource = null;
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = null;
  if (audioContext) {
    const context = audioContext;
    audioContext = null;
    context.close();
  }
}

function addStreamingAnalysisSegment(container, emptyState, sourceText, resultText, className = "") {
  const segment = document.createElement("article");
  segment.className = "streaming-segment";
  const source = document.createElement("p");
  source.className = "streaming-segment-source";
  source.textContent = sourceText;
  const result = document.createElement("p");
  result.className = `streaming-segment-result ${className}`.trim();
  result.textContent = resultText;
  segment.append(source, result);
  container.append(segment);
  emptyState.hidden = true;
}

function renderStreamingAnalysis(message) {
  if (message.translation) {
    addStreamingAnalysisSegment(
      elements.streamingTranslations,
      elements.streamingTranslationEmpty,
      message.source_text,
      message.translation,
    );
  } else if (message.translation_error) {
    addStreamingAnalysisSegment(
      elements.streamingTranslations,
      elements.streamingTranslationEmpty,
      message.source_text,
      `Translation unavailable: ${message.translation_error}`,
      "streaming-segment-error",
    );
  }

  const emotions = Array.isArray(message.emotions) ? message.emotions.slice(0, 3) : [];
  if (emotions.length) {
    const text = emotions
      .map((emotion) => `${emotion.label} ${Math.round(emotion.score * 100)}%`)
      .join(" · ");
    addStreamingAnalysisSegment(
      elements.streamingEmotions,
      elements.streamingEmotionEmpty,
      message.source_text,
      text,
      "streaming-segment-emotion",
    );
  } else if (message.emotion_error) {
    addStreamingAnalysisSegment(
      elements.streamingEmotions,
      elements.streamingEmotionEmpty,
      message.source_text,
      `Emotion estimate unavailable: ${message.emotion_error}`,
      "streaming-segment-error",
    );
  }
}

function addConversationTurn(role, text) {
  const turn = document.createElement("article");
  turn.className = `streaming-turn ${role === "assistant" ? "assistant" : "user"}`;
  const roleLabel = document.createElement("span");
  roleLabel.className = "streaming-turn-role";
  roleLabel.textContent = role === "assistant" ? "Assistant" : "You";
  const bubble = document.createElement("p");
  bubble.className = "streaming-turn-bubble";
  bubble.textContent = text;
  turn.append(roleLabel, bubble);
  elements.streamingConversationLog.append(turn);
  elements.streamingConversationEmpty.hidden = true;
}

function setMicrophoneMuted(muted) {
  if (!microphoneStream) return;
  microphoneStream.getAudioTracks().forEach((track) => {
    track.enabled = !muted && streamingActive;
  });
}

function speakNextAssistantReply(generation) {
  if (generation !== assistantSpeechGeneration) return;
  const text = assistantSpeechQueue.shift();
  if (!text) {
    assistantSpeechActive = false;
    setMicrophoneMuted(false);
    return;
  }
  const voice = localTtsVoices.find((candidate) => candidate.name === elements.ttsVoice.value) || localTtsVoices[0];
  if (!voice) {
    assistantSpeechActive = false;
    setMicrophoneMuted(false);
    elements.streamingStatus.textContent = "No local English voice is available for speech playback.";
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  utterance.onend = () => {
    assistantSpeechActive = false;
    speakNextAssistantReply(generation);
  };
  utterance.onerror = () => {
    if (generation !== assistantSpeechGeneration) return;
    assistantSpeechActive = false;
    elements.streamingStatus.textContent = "The local voice could not play the assistant response.";
    speakNextAssistantReply(generation);
  };
  assistantSpeechActive = true;
  elements.streamingStatus.textContent = "Assistant speaking; microphone paused to prevent feedback.";
  window.speechSynthesis.speak(utterance);
}

function speakAssistantReply(text) {
  if (!text || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return;
  assistantSpeechQueue.push(text);
  setMicrophoneMuted(true);
  if (!assistantSpeechActive) speakNextAssistantReply(assistantSpeechGeneration);
}

function resetAssistantSpeech() {
  assistantSpeechGeneration += 1;
  window.speechSynthesis?.cancel();
  assistantSpeechQueue = [];
  assistantSpeechActive = false;
  setMicrophoneMuted(false);
}

function refreshLocalTtsVoices() {
  if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
    localTtsVoices = [];
    elements.conversationMode.disabled = true;
    elements.ttsVoice.disabled = true;
    elements.conversationHint.textContent = "This browser does not support local speech playback.";
    return;
  }
  const selectedVoice = elements.ttsVoice.value;
  localTtsVoices = window.speechSynthesis.getVoices().filter(
    (voice) => voice.localService && /^en([_-]|$)/i.test(voice.lang),
  );
  elements.ttsVoice.replaceChildren();
  for (const voice of localTtsVoices) {
    const option = document.createElement("option");
    option.value = voice.name;
    option.textContent = `${voice.name} (${voice.lang})`;
    elements.ttsVoice.append(option);
  }
  const preferred = localTtsVoices.find((voice) => voice.name === selectedVoice) || localTtsVoices[0];
  if (preferred) elements.ttsVoice.value = preferred.name;
  const available = localTtsVoices.length > 0;
  elements.conversationMode.disabled = !available || streamingActive;
  elements.ttsVoice.disabled = !available || streamingActive;
  elements.conversationHint.textContent = available
    ? "Replies use this installed on-device English voice; the microphone pauses during playback."
    : "No local English voice was found. Install an English voice in your system settings.";
}

function handleStreamingMessage(event) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch (error) {
    elements.streamingStatus.textContent = "The local transcription service sent an invalid response.";
    return;
  }

  if (message.type === "ready") {
    elements.streamingStatus.textContent = "Listening locally. Speak naturally; interim text appears below.";
    streamingReadyResolve?.();
    streamingReadyResolve = null;
    streamingReadyReject = null;
  } else if (message.type === "configured") {
    elements.streamingTranslationLanguage.textContent = message.target_name || "";
    elements.conversationMode.checked = Boolean(message.conversation_mode);
    elements.streamingConversationPanel.hidden = !message.conversation_mode;
    elements.streamingStatus.textContent = message.conversation_mode
      ? "Listening locally. Phrase analysis and spoken conversation are enabled."
      : "Listening locally. Translation and text emotion update after each recognized phrase.";
    streamingConfigResolve?.();
    streamingConfigResolve = null;
    streamingConfigReject = null;
  } else if (message.type === "partial") {
    elements.streamingText.textContent = message.text || "";
    elements.streamingStatus.textContent = message.endpoint
      ? "Phrase recognized. Listening for more…"
      : "Transcribing live audio…";
  } else if (message.type === "analysis_status") {
    elements.streamingStatus.textContent = "Analyzing the latest phrase locally…";
    if (elements.conversationMode.checked && message.source_text) {
      addConversationTurn("user", message.source_text);
    }
  } else if (message.type === "analysis") {
    renderStreamingAnalysis(message);
    if (message.conversation_error) {
      addConversationTurn("assistant", `Conversation unavailable: ${message.conversation_error}`);
    } else if (message.assistant_response) {
      addConversationTurn("assistant", message.assistant_response);
      speakAssistantReply(message.assistant_response);
    } else {
      elements.streamingStatus.textContent = "Live transcription and phrase analysis are up to date.";
    }
  } else if (message.type === "final") {
    elements.streamingText.textContent = message.text || "";
    elements.streamingStatus.textContent = message.text
      ? "Live transcription, translation, and emotion analysis complete."
      : "No speech was recognized.";
    streamingActive = false;
    elements.targetLanguage.disabled = false;
    elements.conversationMode.disabled = localTtsVoices.length === 0;
    elements.ttsVoice.disabled = localTtsVoices.length === 0;
    setRecordingUi(false);
    stopAudioCapture();
    streamingSocket?.close();
    streamingSocket = null;
  } else if (message.type === "error") {
    elements.streamingStatus.textContent = message.message || "Streaming transcription failed.";
    streamingReadyReject?.(new Error(elements.streamingStatus.textContent));
    streamingConfigReject?.(new Error(elements.streamingStatus.textContent));
    streamingReadyResolve = null;
    streamingReadyReject = null;
    streamingConfigResolve = null;
    streamingConfigReject = null;
    streamingActive = false;
    elements.recordButton.disabled = false;
    elements.targetLanguage.disabled = false;
    elements.conversationMode.disabled = localTtsVoices.length === 0;
    elements.ttsVoice.disabled = localTtsVoices.length === 0;
    resetAssistantSpeech();
    setRecordingUi(false);
    stopAudioCapture();
    streamingSocket?.close();
    streamingSocket = null;
  }
}

function sendPcmChunk(samples) {
  if (!streamingSocket || streamingSocket.readyState !== WebSocket.OPEN) return;
  const resampled = resampleAudio(samples, audioContext.sampleRate);
  if (resampled.length) streamingSocket.send(resampled.buffer);
}

async function startStreamingRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    elements.recordHint.textContent = "Live microphone transcription is not supported in this browser. Upload an audio file instead.";
    return;
  }
  resetAssistantSpeech();
  elements.recordButton.disabled = true;
  elements.targetLanguage.disabled = true;
  elements.conversationMode.disabled = true;
  elements.ttsVoice.disabled = true;
  elements.streamingPanel.hidden = false;
  elements.streamingConversationPanel.hidden = !elements.conversationMode.checked;
  elements.emptyState.hidden = true;
  elements.results.hidden = true;
  elements.loadingState.hidden = true;
  elements.streamingText.textContent = "";
  elements.streamingTranslationLanguage.textContent = elements.targetLanguage.selectedOptions[0]?.textContent || "";
  elements.streamingTranslations.replaceChildren();
  elements.streamingEmotions.replaceChildren();
  elements.streamingConversationLog.replaceChildren();
  elements.streamingTranslationEmpty.hidden = false;
  elements.streamingEmotionEmpty.hidden = false;
  elements.streamingConversationEmpty.hidden = false;
  elements.streamingStatus.textContent = "Waiting for microphone permission…";
  streamingActive = true;

  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    clearSelectedFile();
    elements.streamingPanel.hidden = false;
    elements.emptyState.hidden = true;
    elements.streamingStatus.textContent = "Connecting to the local English streaming model…";

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    streamingSocket = new WebSocket(`${protocol}//${window.location.host}/ws/transcribe`);
    const socket = streamingSocket;
    streamingSocket.binaryType = "arraybuffer";
    streamingSocket.addEventListener("message", (event) => {
      if (streamingSocket === socket) handleStreamingMessage(event);
    });
    streamingSocket.addEventListener("close", () => {
      if (streamingSocket !== socket) return;
      if (!streamingActive) return;
      streamingReadyReject?.(new Error("The streaming connection closed."));
      streamingReadyResolve = null;
      streamingReadyReject = null;
      streamingConfigReject?.(new Error("The streaming connection closed."));
      streamingConfigResolve = null;
      streamingConfigReject = null;
      streamingActive = false;
      elements.recordButton.disabled = false;
      elements.targetLanguage.disabled = false;
      elements.conversationMode.disabled = localTtsVoices.length === 0;
      elements.ttsVoice.disabled = localTtsVoices.length === 0;
      resetAssistantSpeech();
      setRecordingUi(false);
      stopAudioCapture();
      elements.streamingStatus.textContent = "The local streaming connection closed unexpectedly.";
    });
    const readyPromise = new Promise((resolve, reject) => {
      streamingReadyResolve = resolve;
      streamingReadyReject = reject;
      streamingSocket.addEventListener("error", () => reject(new Error("Could not connect to the local streaming service.")), { once: true });
    });
    await readyPromise;
    const configurationPromise = new Promise((resolve, reject) => {
      streamingConfigResolve = resolve;
      streamingConfigReject = reject;
    });
    streamingSocket.send(JSON.stringify({
      type: "configure",
      target_language: elements.targetLanguage.value,
      conversation_mode: elements.conversationMode.checked,
    }));
    await configurationPromise;

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioContextClass({ sampleRate: 16000, latencyHint: "interactive" });
    await audioContext.audioWorklet.addModule("/pcm-capture.js");
    microphoneSource = audioContext.createMediaStreamSource(microphoneStream);
    audioProcessor = new AudioWorkletNode(audioContext, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      outputChannelCount: [1],
    });
    audioProcessor.port.onmessage = (event) => {
      if (event.data.type === "audio") sendPcmChunk(event.data.samples);
      if (event.data.type === "flushed") streamingFlushResolve?.();
    };
    microphoneSource.connect(audioProcessor);
    audioProcessor.connect(audioContext.destination);
    await audioContext.resume();

    elements.recordButton.disabled = false;
    setRecordingUi(true);
    elements.recordHint.textContent = elements.conversationMode.checked
      ? "Speak a phrase, then pause for the assistant to answer. The microphone pauses while it speaks."
      : "Streaming English speech; translation and text-emotion estimates update by phrase.";
    recordingStartedAt = Date.now();
    elements.recordTimer = window.setInterval(() => {
      elements.recordTime.textContent = formatDuration((Date.now() - recordingStartedAt) / 1000);
    }, 1000);
  } catch (error) {
    streamingActive = false;
    elements.recordButton.disabled = false;
    elements.targetLanguage.disabled = false;
    elements.conversationMode.disabled = localTtsVoices.length === 0;
    elements.ttsVoice.disabled = localTtsVoices.length === 0;
    streamingReadyResolve = null;
    streamingReadyReject = null;
    streamingConfigResolve = null;
    streamingConfigReject = null;
    resetAssistantSpeech();
    setRecordingUi(false);
    stopAudioCapture();
    if (streamingSocket && streamingSocket.readyState < WebSocket.CLOSING) streamingSocket.close();
    streamingSocket = null;
    elements.streamingStatus.textContent = error.message || "Could not start live transcription.";
    elements.recordHint.textContent = error.name === "NotAllowedError"
      ? "Microphone access was denied. Allow it in browser settings or upload a file."
      : "Live transcription could not start. Check the local streaming model and try again.";
  }
}

async function stopStreamingRecording() {
  if (!streamingActive) return;
  elements.recordButton.disabled = true;
  elements.streamingStatus.textContent = "Finishing the current utterance…";
  if (audioProcessor && streamingSocket?.readyState === WebSocket.OPEN) {
    const flushPromise = new Promise((resolve) => {
      streamingFlushResolve = resolve;
      window.setTimeout(resolve, 500);
    });
    audioProcessor.port.postMessage({ type: "flush" });
    await flushPromise;
    streamingFlushResolve = null;
    streamingSocket.send(JSON.stringify({ type: "end" }));
  } else if (streamingSocket?.readyState === WebSocket.OPEN) {
    streamingSocket.send(JSON.stringify({ type: "end" }));
  }
  streamingActive = false;
  elements.recordButton.disabled = false;
  setRecordingUi(false);
  stopAudioCapture();
  elements.streamingStatus.textContent = "Finalizing the local transcript…";
}

async function toggleRecording() {
  if (streamingActive) await stopStreamingRecording();
  else await startStreamingRecording();
}

function renderList(container, items, className = "") {
  container.innerHTML = items.map((item) => `<span class="tag ${className}">${escapeHtml(item)}</span>`).join("");
}

function renderBullets(container, items) {
  container.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function showInlineError(selector, message) {
  const element = document.querySelector(selector);
  element.textContent = message || "";
  element.hidden = !message;
}

function renderResults(data) {
  const errors = Object.entries(data.errors || {}).filter(([, message]) => message);
  elements.analysisErrors.innerHTML = errors.length
    ? `<strong>Some readouts are unavailable.</strong><ul>${errors.map(([stage, message]) => `<li><strong>${escapeHtml(stage.replaceAll("_", " "))}:</strong> ${escapeHtml(message)}</li>`).join("")}</ul>`
    : "";
  elements.analysisErrors.hidden = errors.length === 0;
  const agents = Array.isArray(data.agents) ? data.agents : [];
  elements.agentActivity.innerHTML = agents.map((agent) => {
    const status = ["complete", "warning", "error", "skipped"].includes(agent.status) ? agent.status : "unknown";
    return `<li class="agent-event">
      <div class="agent-event-heading">
        <div><strong>${escapeHtml(agent.name)}</strong><span>${escapeHtml(agent.role)}</span></div>
        <span class="agent-status status-${status}">${escapeHtml(status)}</span>
      </div>
      <p>${escapeHtml(agent.detail)}</p>
    </li>`;
  }).join("");
  const reviewNotes = Array.isArray(data.review_notes) ? data.review_notes : [];
  elements.agentReviewNotes.textContent = reviewNotes.join(" ");
  elements.agentReviewNotes.hidden = reviewNotes.length === 0;

  const metrics = data.metrics || {};
  document.querySelector("#metric-duration").textContent = formatDuration(metrics.duration_seconds ?? audioDuration);
  document.querySelector("#metric-pace").textContent = metrics.words_per_minute == null ? "—" : metrics.words_per_minute;

  const emotionList = Array.isArray(data.emotions) ? data.emotions : [];
  const leadingEmotion = emotionList[0];
  document.querySelector("#metric-emotion").textContent = leadingEmotion?.label || "—";
  document.querySelector("#metric-emotion-confidence").textContent = leadingEmotion ? `${Math.round(leadingEmotion.score * 100)}% model confidence` : "vocal signal";
  document.querySelector("#emotion-list").innerHTML = emotionList.map((emotion) => {
    const score = Math.max(0, Math.min(1, Number(emotion.score) || 0));
    return `<div class="emotion-row"><span class="emotion-name">${escapeHtml(emotion.label)}</span><span class="emotion-track"><span class="emotion-fill" style="width:${Math.round(score * 100)}%"></span></span><span class="emotion-score">${Math.round(score * 100)}%</span></div>`;
  }).join("");
  showInlineError("#emotion-error", data.errors?.emotion);

  const language = data.detected_language;
  document.querySelector("#metric-language").textContent = language?.name || "—";
  document.querySelector("#metric-language-confidence").textContent = language ? `${Math.round(language.confidence * 100)}% confidence` : "detected";
  document.querySelector("#transcript-language").textContent = language?.name || "Language unavailable";

  currentTranscript = data.transcript || "";
  document.querySelector("#transcript-text").textContent = currentTranscript;
  const wordCount = metrics.word_count ?? currentTranscript.split(/\s+/).filter(Boolean).length;
  document.querySelector("#transcript-word-count").textContent = `${wordCount} ${wordCount === 1 ? "word" : "words"}`;
  showInlineError("#transcript-error", data.errors?.transcription);

  const translation = data.translation;
  document.querySelector("#translation-language").textContent = elements.targetLanguage.selectedOptions[0]?.textContent || "";
  document.querySelector("#translation-text").textContent = translation || "";
  document.querySelector("#translation-text").hidden = !translation;
  showInlineError("#translation-error", data.errors?.translation);

  const analysis = data.analysis;
  document.querySelector("#summary-text").textContent = analysis?.summary || "";
  document.querySelector("#summary-text").hidden = !analysis?.summary;
  const sentimentPill = document.querySelector("#sentiment-pill");
  sentimentPill.textContent = analysis?.sentiment
    ? `${analysis.sentiment} · ${Number(analysis.sentiment_score).toFixed(2)}`
    : "";
  sentimentPill.hidden = !analysis?.sentiment;
  const toneRow = document.querySelector("#tone-row");
  document.querySelector("#tone-value").textContent = analysis?.tone || "";
  toneRow.hidden = !analysis?.tone;

  const themes = [...(analysis?.themes || []), ...(analysis?.key_phrases || [])];
  renderList(document.querySelector("#theme-list"), themes.slice(0, 12));
  document.querySelector("#theme-list").hidden = themes.length === 0;
  document.querySelector("#themes-empty").hidden = themes.length !== 0;
  renderBullets(document.querySelector("#action-list"), analysis?.action_items || []);
  document.querySelector("#action-list").hidden = !(analysis?.action_items?.length);
  document.querySelector("#actions-empty").hidden = Boolean(analysis?.action_items?.length);
  renderBullets(document.querySelector("#communication-list"), analysis?.communication_notes || []);
  document.querySelector("#communication-list").hidden = !(analysis?.communication_notes?.length);
  document.querySelector("#communication-empty").hidden = Boolean(analysis?.communication_notes?.length);

  showInlineError("#summary-text", data.errors?.analysis);
  document.querySelector("#summary-text").hidden = Boolean(data.errors?.analysis);
}

function rotateLoadingMessage() {
  const messages = [
    ["Transcribing your audio", "The transcription agent is identifying speech and language…"],
    ["Specialists at work", "Emotion, insights, and translation agents are handling their assignments…"],
    ["Reviewing the readout", "The reviewer checks outputs before the metrics agent finishes…"],
  ];
  let index = 0;
  elements.loadingTitle.textContent = messages[index][0];
  elements.loadingDescription.textContent = messages[index][1];
  loadingTimer = window.setInterval(() => {
    index = (index + 1) % messages.length;
    elements.loadingTitle.textContent = messages[index][0];
    elements.loadingDescription.textContent = messages[index][1];
    document.querySelectorAll(".pipeline-step").forEach((step, stepIndex) => {
      step.classList.toggle("active", stepIndex <= index);
    });
  }, 6500);
}

async function analyzeAudio() {
  if (!selectedFile) return;
  elements.analyzeButton.disabled = true;
  elements.emptyState.hidden = true;
  elements.results.hidden = true;
  elements.loadingState.hidden = false;
  elements.configurationNotice.hidden = true;
  document.querySelectorAll(".pipeline-step").forEach((step, index) => step.classList.toggle("active", index === 0));
  rotateLoadingMessage();

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("target_language", elements.targetLanguage.value);
  const duration = audioDuration ?? elements.preview.duration;
  if (Number.isFinite(duration) && duration > 0) formData.append("duration_seconds", String(duration));
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed. Please try again.");
    renderResults(data);
    elements.loadingState.hidden = true;
    elements.results.hidden = false;
  } catch (error) {
    elements.loadingState.hidden = true;
    elements.emptyState.hidden = false;
    showNotice(error.message || "Something went wrong while analyzing this recording.", "notice-error");
  } finally {
    window.clearInterval(loadingTimer);
    elements.analyzeButton.disabled = !selectedFile;
  }
}

elements.fileInput.addEventListener("change", (event) => setSelectedFile(event.target.files?.[0]));
elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("drag-over");
});
elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("drag-over"));
elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("drag-over");
  setSelectedFile(event.dataTransfer.files?.[0]);
});
elements.recordButton.addEventListener("click", toggleRecording);
elements.removeAudio.addEventListener("click", clearSelectedFile);
elements.analyzeButton.addEventListener("click", analyzeAudio);
elements.newAnalysis.addEventListener("click", clearSelectedFile);
elements.conversationMode.addEventListener("change", () => {
  elements.streamingConversationPanel.hidden = !elements.conversationMode.checked;
});
document.querySelector("#copy-transcript").addEventListener("click", async (event) => {
  if (!currentTranscript) return;
  try {
    await navigator.clipboard.writeText(currentTranscript);
    event.currentTarget.textContent = "Copied";
    window.setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1500);
  } catch {
    event.currentTarget.textContent = "Copy unavailable";
  }
});

refreshLocalTtsVoices();
window.speechSynthesis?.addEventListener("voiceschanged", refreshLocalTtsVoices);

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    elements.recordButton.disabled = !health.streaming_asr_ready;
    if (!health.streaming_asr_ready) {
      showNotice("The local streaming ASR model is missing. Run .venv\\Scripts\\python.exe server\\download_models.py to download it once.");
    } else if (!health.local_models_ready) {
      showNotice("Local speech models are missing. Run .venv\\Scripts\\python.exe server\\download_models.py to download them once.");
    } else if (!health.local_llm_ready) {
      showNotice("The local speech models are ready. Start a loopback llama.cpp server with run-server.ps1 or run-server-small.ps1 to enable summaries and translation.");
    }
  })
  .catch(() => showNotice("The local app server could not be reached. Start it with run-speech-analytics.ps1 and refresh this page.", "notice-error"));
