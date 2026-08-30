import sip_whisper
from pathlib import Path
import torch

model = sip_whisper.load_model("large-v3-turbo", device=torch.device("cuda"))
audio_path = Path.cwd()/"SNR40_s21_bgir5s.wav"
audio = sip_whisper.load_audio(str(audio_path))
audio = sip_whisper.pad_or_trim(audio)

options = {
    "model": model,
    "audio": audio,
    "fp16": False,
    "task": "transcribe",
    "beam_size": 5,
    "temperature": 0,
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "language": "en",
    "subword_timestamps": False,
    "forced_alignment_options": {"position": 1, "token_id_or_word": " test"}
}
result = sip_whisper.transcribe(**options)
print(result["text"])