import pytest
import sip_whisper
import whisper
import torch

@pytest.mark.parametrize("time_stamps", [False,True])
def test_sip_whisper_module(time_stamps):
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(model,
                       audio,
                       fp16=False,
                       beam_size=2,
                       temperature=0,
                       word_timestamps=time_stamps,
                       condition_on_previous_text=False)

    result["extracted_logprobs"]
    print()

@pytest.mark.parametrize("file_path", ["tests/sample_audio_small.mp3","tests/testfile.wav"])
def test_compare_sip_whisper_with_original(file_path):
    #sip_whisper
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio(file_path, 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result_1 = sip_whisper.transcribe(model,
                                      audio,
                                      fp16=False,
                                      beam_size=2,
                                      temperature=0,
                                      word_timestamps=True,
                                      condition_on_previous_text=False)
    tensors = result_1.pop("extracted_logprobs")

    # regular whisper
    model = whisper.load_model("tiny", device="cpu")
    audio = whisper.load_audio(file_path, 16_000)
    audio = whisper.pad_or_trim(audio)

    result_2 = whisper.transcribe(model,
                                      audio,
                                      fp16=False,
                                      beam_size=2,
                                      temperature=0,
                                      word_timestamps=True,
                                      condition_on_previous_text=False)
    assert result_1 == result_2

@pytest.mark.parametrize(("file_path", "result_tensor"), [
    ("tests/sample_audio_small.mp3", "tests/sip_result_sample_audio_small.mp3_.pt"),
    ("tests/testfile.wav", "tests/sip_result_testfile.wav_.pt"),
    ])
def test_consistency(file_path, result_tensor):
    #sip_whisper
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio(file_path, 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(model,
                                      audio,
                                      fp16=False,
                                      beam_size=5,
                                      temperature=0,
                                      word_timestamps=True,
                                      condition_on_previous_text=False)
    sip_result = result.pop("extracted_logprobs")
    assert torch.equal(sip_result, torch.load(result_tensor))


def test_mixing_functions():
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result_1 = sip_whisper.transcribe(model,
                                    audio,
                                    fp16=False,
                                    beam_size=2,
                                    temperature=0,
                                    word_timestamps=True,
                                    condition_on_previous_text=False)
    tensors_1 = result_1.pop("extracted_logprobs")


    #use regular whisper functions
    audio_2 = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio_2 = whisper.pad_or_trim(audio_2)

    result_2 = sip_whisper.transcribe(model,
                                    audio_2,
                                    fp16=False,
                                    beam_size=2,
                                    temperature=0,
                                    word_timestamps=True,
                                    condition_on_previous_text=False)
    tensors_2 = result_2.pop("extracted_logprobs")

    assert result_1 == result_2
    assert torch.equal(tensors_1, tensors_2)
    print()

def test_proof_determinism():
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    sip_results = []
    for i in range(5):


        result = sip_whisper.transcribe(model,
                                          audio,
                                          fp16=False,
                                          beam_size=2,
                                          temperature=0,
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

    result = sip_whisper.transcribe(model,
                                    audio,
                                    fp16=False,
                                    beam_size=5,
                                    temperature=0,
                                    word_timestamps=True,
                                    condition_on_previous_text=False
                                    )
    print(result)


