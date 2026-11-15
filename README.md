# agentic-speech-analytics-conversation

Agentic speech analytics and spoken language conversation.

## Signal Speech Analytics + Muse Glimmer (local, this laptop)

Signal is a local speech analytics site for microphone recordings and uploaded
audio. A LangGraph supervisor routes each transcript through specialist
transcription, emotion, insights, translation, review, and metrics agents. The
agents use local Faster-Whisper, DistilRoBERTa, and the llama.cpp model server.
The workflow is deliberately sequential after transcription to avoid loading
multiple memory-heavy models at once on this laptop.

Microphone capture uses Sherpa-ONNX Zipformer English 20M for incremental
streaming transcription over a local WebSocket. The browser sends 16 kHz audio
frames to the local app and displays partial text while speaking. At each
recognized utterance boundary, local Qwen translates that phrase and the local
DistilRoBERTa classifier estimates emotion from its words; those results are
appended incrementally. Translation language is selected before microphone
capture. This streaming model is English-only, and emotion estimates are based
on transcript wording rather than vocal prosody. Full summaries, themes, action
items, and other LangGraph readouts remain part of uploaded-audio analysis.

Optional spoken-conversation mode sends each recognized phrase to the local
llama.cpp model with a bounded six-turn context and returns a short spoken
reply. Qwen3-TTS 0.6B synthesizes each reply locally using the selected Aiden or
Ryan voice; the browser only plays audio returned by the local app. Microphone
capture pauses during playback to prevent acoustic feedback. Conversation
history is kept in memory only for the active WebSocket session, and generated
audio is not stored. The CPU implementation synthesizes a complete WAV before
playback; on this machine the first short test took about 22 seconds including
model load, and a warm 3.7-second sample took about 11 seconds to generate.
Synthesis is slower than real time, so the microphone stays paused during
generation and playback.

The results include an agent activity trace showing each specialist's
assignment, completion status, and any skipped or warning steps. The emotion
agent is conditionally skipped unless language detection identifies English
with at least 70% confidence. The review agent applies deterministic output
checks and screens generated action items; it does not fact-check model claims.
The LangGraph path is `START`, `transcription_agent`, `supervisor`, the
optional `emotion_agent`, `insights_agent`, `translation_agent`, `review_agent`,
`metrics_agent`, then `END`. Specialist nodes run sequentially to limit peak
memory.

**Inference is offline:** the app sets Hugging Face/Transformers offline flags,
loads model weights from `models\`, disables HTTP proxy inheritance, and only
allows the language-model endpoint at `127.0.0.1`. Audio and transcripts are not
sent to Hugging Face or another remote inference service. The one-time model
download step below fetches public model files only; it does not upload user
recordings. Model license terms are in the model cards.

## Set up and run the speech analytics site

1. Install Python 3.12. If `py` is not available in a new terminal after
   installation, use the Python executable from your installation directory.
2. From the repository root, create the virtual environment and install the
   CPU-only PyTorch build plus the app dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   .\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
   Copy-Item server\.env.example server\.env
   .\.venv\Scripts\python.exe server\download_models.py
   ```

   This downloads Faster-Whisper small, the English emotion classifier,
   streaming Zipformer, Qwen3-TTS 0.6B CustomVoice, and a small Qwen GGUF used
   for local smoke tests into the ignored `models\` directory. Runtime
   inference never downloads weights. Qwen TTS runs on CPU using bfloat16 and
   can take longer to synthesize than browser-native speech.
3. Start the local language model server in one terminal:

   ```powershell
   .\run-server.ps1
   ```

   The existing Muse Glimmer 30B model needs substantial RAM. If it does not
   fit, use the included small Qwen model on port 8081 and set
   `LLM_ENDPOINT=http://127.0.0.1:8081/v1/chat/completions` and
   `LLM_MODEL=qwen-local` in `server\.env`.
4. Start the web app in another terminal:

   ```powershell
   .\run-speech-analytics.ps1
   ```

5. Open <http://127.0.0.1:8000>.

Microphone capture requires browser microphone permission and is available on
localhost. A live session is limited to 30 minutes of audio and closes after
60 seconds without audio. Uploaded audio is limited to 25 MB and is processed
locally. The emotion model is English-only and estimates emotion from transcript
wording, not vocal tone. The app displays per-step errors instead of fabricating
results when a local model is unavailable. Spoken conversation requires the
downloaded local Qwen TTS model; it does not depend on browser-installed voices.
Run the defensive checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest server\tests
```

### Low-resource launch without PowerShell scripts

If this computer's execution policy blocks `.ps1` files, keep that policy
unchanged and launch the local executables directly. In one terminal:

```powershell
.\runtime\llama.cpp\llama-server.exe -m .\models\speech-analytics\qwen2.5-0.5b-instruct\qwen2.5-0.5b-instruct-q4_k_m.gguf -a qwen-local -ngl 0 -c 2048 -np 1 --host 127.0.0.1 --port 8081
```

In another terminal:

```powershell
$env:LLM_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"
$env:LLM_MODEL = "qwen-local"
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir server --host 127.0.0.1 --port 8000
```

### Validation and model limitations

The local smoke test used a synthetic 12-second English clip containing a
spoken prompt-injection phrase. Faster-Whisper small transcribed the clip and
detected English at 99.8% confidence; the transcript-based emotion classifier
ranked neutral first at 96%. This is a single functional test, not an accuracy
benchmark, and the emotion model does not analyze vocal prosody.

The Qwen 0.5B model produced a poor Spanish translation and initially promoted
the spoken injection into action items. The review agent filters common
English model-directed and credential-disclosure action items, but that
heuristic is not a general prompt-injection detector. Treat all generated
insights and translations as unverified; the 30B model has not been evaluated
here, and neither model is suitable for consequential decisions without human
review.

## Muse Glimmer local chat (optional)

Meta **Muse Glimmer 30B** prepared for a **16 GB** AMD Ryzen AI 7 350 laptop (Radeon 860M, no NVIDIA GPU).

The official 17 GB Q4 GGUF does not fit in 16 GB RAM. This folder uses Unsloth **UD-IQ2_XXS** (~10 GB) plus llama.cpp **Vulkan** (`b10549`).

## Run

1. Close heavy apps (browser tabs, other models).
2. Start the server:

```powershell
.\run-server.ps1
```

3. In another terminal, smoke-test:

```powershell
.\chat.ps1
```

OpenAI-compatible API: `http://127.0.0.1:8080/v1/chat/completions`, model name `muse-glimmer`.

## Hardware notes

| Item | This machine |
| --- | --- |
| RAM | 16 GB — tight; keep context at 4096 |
| GPU | Radeon 860M via Vulkan, not CUDA |
| Speed | Usable but slow vs a 24 GB NVIDIA card |
| Vision | Not loaded (mmproj would add ~1.4 GB) |

If load fails with out-of-memory, reboot, close apps, or drop `-c` in `run-server.ps1` to `2048`. If Vulkan misbehaves, change `-ngl 99` to `-ngl 0` for CPU-only.

Apache 2.0 weights: [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF). Official Meta files: [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF).
