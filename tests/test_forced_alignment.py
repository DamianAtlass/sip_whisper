import pytest
import sip_whisper
import torch
from pathlib import Path

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
clean_audio_path = test_folder / "SNR40_s21_bgir5s.wav"

@pytest.mark.parametrize(("forced_alignment_options", "result_text"), (
        #[None, " been green in R5 soon."],
        #[{"position": 1, "token_id_or_word": 1500}, " been testable in R5 soon."], #30000  = ' test'
        #[{"position": 3, "token_id_or_word": 30000}, " been green in kilos five soon."], #30000  = ' kilos'
        [{"position": 1, "token_id_or_word": " blue"}, " been blue in R5 soon."],  # 30000  = ' kilos'
))
def test_forced_alignment(forced_alignment_options: dict, result_text):
    model = sip_whisper.load_model("large-v3-turbo", device=torch.device("cuda"))
    audio_path = Path.cwd() / "SNR40_s21_bgir5s.wav"
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
        "subword_timestamps": True,
        "forced_alignment_options": forced_alignment_options # 1500 = " test"
    }
    result = sip_whisper.transcribe(**options)
    assert result["text"] == result_text
    if forced_alignment_options is not None and isinstance(forced_alignment_options["token_id_or_word"], int):
        assert result["segments"][0]["tokens"][forced_alignment_options["position"]] == forced_alignment_options["token_id_or_word"]
    print()