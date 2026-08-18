import pytest
import sip_whisper
import whisper
import torch
from pathlib import Path

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()

@pytest.mark.parametrize("time_stamps", [False,True])
def test_sip_whisper_module(time_stamps):
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=2,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=time_stamps,
        condition_on_previous_text=False)

    result["extracted_logprobs"]
    print()

@pytest.mark.parametrize("file_path", ["tests/sample_audio_small.mp3"])#,"tests/testfile.wav"])
def test_compare_sip_whisper_with_original(file_path):
    #sip_whisper
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio(file_path, 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result_1 = sip_whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=2,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=True,
        condition_on_previous_text=False)
    tensors = result_1.pop("extracted_logprobs")

    # regular whisper
    model = whisper.load_model("tiny", device="cpu")
    audio = whisper.load_audio(file_path, 16_000)
    audio = whisper.pad_or_trim(audio)

    result_2 = whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=2,
        temperature=0,
        word_timestamps=True,
        condition_on_previous_text=False)

    assert result_1["segments"][0]["tokens"] == result_2["segments"][0]["tokens"]

@pytest.mark.parametrize(("file_path", "tensor_path"), [
    ("sample_audio_small.mp3", "tensor_sample_audio_small.mp3.pt"),
    ("testfile.wav", "tensor_testfile.wav.pt"), #hashed may vary depending on how the GPU that calculated the tensors
    ])
def test_consistency(file_path, tensor_path):
    file_path = test_folder / file_path
    tensor_path = test_folder / tensor_path
    model = sip_whisper.load_model("tiny", device="cuda")
    audio = sip_whisper.load_audio(file_path, 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=5,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=True,
        condition_on_previous_text=False)
    sip_result = result.pop("extracted_logprobs")
    if not tensor_path.exists():
        torch.save(sip_result, tensor_path)
        pytest.skip()
    tokens = [x for s in result["segments"] for x in s["tokens"]]
    assert len(tokens) == sip_result.shape[0]
    assert torch.allclose(sip_result, torch.load(tensor_path))


def test_mixing_functions():
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result_1 = sip_whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=2,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=True,
        condition_on_previous_text=False)
    tensors_1 = result_1.pop("extracted_logprobs")


    #use regular whisper functions
    audio_2 = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio_2 = whisper.pad_or_trim(audio_2)

    result_2 = sip_whisper.transcribe(
        model,
        audio_2,
        fp16=False,
        beam_size=2,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=True,
        condition_on_previous_text=False)
    tensors_2 = result_2.pop("extracted_logprobs")

    assert result_1["text"] == result_2["text"]
    assert result_1["decoded_tokens_with_timestamps"] == result_2["decoded_tokens_with_timestamps"]
    assert torch.allclose(tensors_1, tensors_2)
    print()

def test_proof_determinism():
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    sip_results = []
    for i in range(5):


        result = sip_whisper.transcribe(
            model,
            audio,
            fp16=False,
            beam_size=2,
            temperature=0,
            extract_logprobs=True,
            word_timestamps=True,
            condition_on_previous_text=False)
        sip_result = result["extracted_logprobs"]
        sip_results.append(sip_result)

    for i in range(1, len(sip_results)):
        torch.equal(sip_results[0], sip_results[i])


def test_difficult_audio():
    # this transcription will recognize the language falsely as welsh, leading to only 4 hypotheses which have an EOT token.
    # The last spot (beamsize is 5!) will be filled up with a very long (and bad) hypothesis, with will be weighted as top hypothesis afterward.

    model = sip_whisper.load_model("tiny")

    audio = sip_whisper.load_audio("tests/testfile.wav", 16_000)

    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(
        model,
        audio,
        fp16=False,
        beam_size=5,
        temperature=0,
        word_timestamps=True,
        condition_on_previous_text=False)
    print(result)

@pytest.mark.parametrize(("subword_timestamps", "expected_length"), [
    (False, 9),
    (True, 11)])
def test_subword_timestamps(subword_timestamps, expected_length):
    model = sip_whisper.load_model("base", device=torch.device("cuda"))
    audio_path = Path.cwd() / "tests" / "s29_bgwszp.wav"
    audio = sip_whisper.load_audio(str(audio_path))
    audio = sip_whisper.pad_or_trim(audio)

    options = {
        "model": model,
        "audio": audio,
        "fp16": False,
        "beam_size": 5,
        "temperature": 0,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "language": "en",
        "subword_timestamps": subword_timestamps,
    }
    result = sip_whisper.transcribe(**options)
    assert len(result["segments"][0]["words"]) == expected_length
    print()
