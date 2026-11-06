"""Qwen2-Audio speech-to-text backend."""
import logging
import os

from .base_stt import TARGET_SAMPLE_RATE, BaseSTT, load_audio

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "ml": "Malayalam",
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
}


def _transcription_prompt(language: str) -> str:
    language_name = LANGUAGE_NAMES.get(language.lower(), language)
    return f"Transcribe the audio in {language_name}. Return only the spoken words."


def _max_new_tokens() -> int:
    try:
        return max(1, int(os.getenv("QWEN_AUDIO_MAX_NEW_TOKENS", "256")))
    except ValueError:
        return 256


class QwenAudioSTT(BaseSTT):
    name = "qwen_audio"

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        configured_id = (model_id or os.getenv(
            "QWEN_AUDIO_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct"
        )).strip()
        expanded_id = os.path.expanduser(configured_id)
        self.model_source = expanded_id
        self.invalid_model_path = (
            expanded_id
            if (os.path.isabs(expanded_id) or expanded_id.startswith("."))
            and not os.path.isdir(expanded_id)
            else ""
        )
        self.max_new_tokens = _max_new_tokens()
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self.invalid_model_path:
            raise FileNotFoundError(f"QWEN_AUDIO_MODEL_ID does not exist: {self.invalid_model_path}")

        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        local_files_only = os.path.isdir(self.model_source)
        self._processor = AutoProcessor.from_pretrained(
            self.model_source,
            local_files_only=local_files_only,
        )
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_source,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        logger.info("Qwen2-Audio '%s' loaded", self.model_source)

    def _transcribe(self, audio_path: str, language: str) -> str:
        audio = load_audio(audio_path, sample_rate=TARGET_SAMPLE_RATE)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": _transcription_prompt(language)},
            ],
        }]
        prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=prompt,
            audios=[audio],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]