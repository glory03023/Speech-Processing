import argparse
import torch
import torchaudio
import whisper
import pyannote
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def get_words_timestamps(result_transcription: dict) -> dict:
    """Get all words their start and end times into a dict"""
    words = []
    for segment in result_transcription["segments"]:
        for word in segment["words"]:
            words.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
            })
    return words


def words_per_segment(
    res_transcription: dict,
    res_diarization: pyannote.core.Annotation,
) -> dict:
    """Get all words per segment and their start and end times into a dict

    Args:
        res_transcription (dict): The transcription result from the whisper library
        res_diarization (pyannote.core.Annotation): The diarization result from the pyannote library

    Returns:
        dict: A dict containing all words per segment and their start and end times and the speaker
    """

    words = get_words_timestamps(res_transcription)
    dia_list = list(res_diarization.itertracks(yield_label=True))
    segments = []
    iS = 0
    nS = len(dia_list)

    while iS < nS:
        jS = iS + 1
        while jS < nS and dia_list[iS][2] == dia_list[jS][2]:
            jS = jS + 1
        segments.append({
            "stop" : dia_list[jS][0].start if jS < nS else words[-1]['end'],
            "speaker" : dia_list[iS][2]
            })
        iS = jS

    res_trans_dia = []

    iS = 0
    nS = len(segments)
    iW = 0
    nW = len(words)

    start = 0
    while iS<nS or iW < nW:
        end = segments[iS]['stop']
        segment_words = []
        while iW < nW and words[iW]['start'] + 2 * words[iW]['end'] <= 3 * end:
        # while iW < nW and words[iW]['end'] <= end:
            segment_words.append(words[iW]['word'])
            iW = iW + 1
        res_trans_dia.append({
            "speaker": segments[iS]['speaker'],
            "text": " ".join(segment_words),
            "start": start,
            "end": end,
        })
        start = end
        iS = iS + 1


    return res_trans_dia

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asrmodel", default="small.en.pt", help="Model for speech recocgnition")
    parser.add_argument("--diarizmodel", default="pyannote/speaker-diarization-3.1", help="Model for speaker diarization")
    parser.add_argument("--hftoken", default="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", help="Hugging face hub token")
    parser.add_argument("--audio", default="test3001.wav", help="audio file to syllable")
 
    args = parser.parse_args()

    print(args)

    pipeline = Pipeline.from_pretrained(
        args.diarizmodel, use_auth_token=args.hftoken
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline.to(device)

    audio_path = args.audio

    waveform, sample_rate = torchaudio.load(audio_path)
    with ProgressHook() as hook:
        diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    print(diarization_result)

    model = whisper.load_model(args.asrmodel)

    print("loaded Whisper model")

    transcription_result = model.transcribe(audio_path, word_timestamps=True)

    print(transcription_result)

    final_result = words_per_segment(transcription_result, diarization_result)

    for segment in final_result:
        print(
            f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t {segment["speaker"]}\t{segment["text"]}'
        )

if __name__ == "__main__":
    main()    