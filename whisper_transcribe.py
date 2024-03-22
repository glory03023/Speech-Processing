#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small.en.pt", help="Model to use")
    parser.add_argument("--audio", default="New Recording 340.mp3", help="audio file to syllable")

    args = parser.parse_args()

    print(args)
    model = args.model
    audio_model = whisper.load_model(model)

    # Load the audio file
    audio_file = args.audio

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    # Read the transcription.
    result = audio_model.transcribe(audio_file, fp16=torch.cuda.is_available(), word_timestamps=True)

    print(result)

if __name__ == "__main__":
    main()