import time
import numpy as np
import uvicorn
import io
import librosa
import zhconv

from fastapi import FastAPI, File, Form, UploadFile
from whisper_model import WhisperModel

app = FastAPI()

whisper = WhisperModel(model_name="small.en.pt")

@app.post('/audioToText')
def audio_to_text(
    timestamp: str = Form(),
    audio: UploadFile = File()
):
    bt = audio.file.read()
    memory_file = io.BytesIO(bt)
    data, sample_rate = librosa.load(memory_file)

    resample_data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

    transcribe_start_time = time.time()
    text = whisper.transcribe(resample_data)
    transcribe_end_time = time.time()

    convert_start_time = time.time()
    text = zhconv.convert(text, 'zh-hans')
    convert_end_time = time.time()

    print(text)

    return {
        'status': 'ok',
        'text': text,
        'transcribe_time': transcribe_end_time - transcribe_start_time,
        'convert_time': convert_end_time - convert_start_time
    }

uvicorn.run(app, host="0.0.0.0", port=9090)