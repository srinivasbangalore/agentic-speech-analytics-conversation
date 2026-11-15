import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "speech-analytics"

MODELS = (
    (
        "Systran/faster-whisper-small",
        "faster-whisper-small",
        ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
    ),
    (
        "j-hartmann/emotion-english-distilroberta-base",
        "emotion-english-distilroberta-base",
        [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    ),
    (
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        [
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ],
    ),
)


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, directory, allow_patterns in MODELS:
        destination = MODEL_ROOT / directory
        print(f"Downloading public model files: {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            allow_patterns=allow_patterns,
            token=False,
        )

    qwen_dir = MODEL_ROOT / "qwen2.5-0.5b-instruct"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading small local GGUF test model -> {qwen_dir}")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        local_dir=str(qwen_dir),
        token=False,
    )
    print("Model downloads complete. Runtime inference is configured for offline/local-only use.")


if __name__ == "__main__":
    main()
