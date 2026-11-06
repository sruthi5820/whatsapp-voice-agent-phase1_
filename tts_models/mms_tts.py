"""Facebook MMS TTS for Malayalam (`facebook/mms-tts-mal`), a VITS-based model."""
import logging
import os

from .base_tts import BaseTTS, SynthesisError, SynthesisResult

logger = logging.getLogger(__name__)


class MMSTTS(BaseTTS):
    name = "mms"

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        self.model_id = model_id or os.getenv("MMS_TTS_MODEL_ID", "facebook/mms-tts-mal")
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = VitsModel.from_pretrained(self.model_id).to(self._device).eval()

    def _synthesize(self, text: str) -> SynthesisResult:
        import torch

        inputs = self._tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].numel() == 0:
            raise SynthesisError("The text contains no characters the model can speak.")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            waveform = self._model(**inputs).waveform
        audio = waveform.squeeze().cpu().numpy().astype("float32")
        return SynthesisResult(waveform=audio, sampling_rate=int(self._model.config.sampling_rate))
