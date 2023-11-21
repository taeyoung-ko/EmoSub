from record import AudioRecorder
from emotion import speechEmotionRecognition
from transcribe import SpeechToText

import asyncio
import websockets
import json

import concurrent.futures


def record_audio(recorder):
    recorded_frames = recorder.record_audio()
    filtered_frames = recorder.filter_noise(recorded_frames)
    recorder.save_audio(filtered_frames)

def emotion_task():
    major_emotion = speechEmotionRecognition.emotion_task("recorded_audio.wav")
    return major_emotion

def transcribe_task(whisper_decoder):
    result_text = whisper_decoder.decode_audio("recorded_audio.wav")
    return result_text

async def send_data(websocket, path):
    recorder = AudioRecorder()
    whisper_decoder = SpeechToText()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            try:
                record_future = executor.submit(record_audio, recorder)

                emotion_future = executor.submit(emotion_task)
                transcribe_future = executor.submit(transcribe_task, whisper_decoder)

                record_result = record_future.result()

                major_emotion = emotion_future.result()
                text = transcribe_future.result()

                data = {'emotion': major_emotion, 'transcription': text}
                print(data)

                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                break


start_server = websockets.serve(send_data, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
