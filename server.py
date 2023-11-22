from record import AudioRecorder
from emotion import speechEmotionRecognition
from transcribe import SpeechToText

import asyncio
import websockets
import json

import concurrent.futures

async def record_audio(recorder):
    recorded_frames = recorder.record_audio()
    filtered_frames = recorder.filter_noise(recorded_frames)
    recorder.save_audio(filtered_frames)

async def emotion_task():
    major_emotion = speechEmotionRecognition.emotion_task("recorded_audio.wav")
    return major_emotion

async def transcribe_task(whisper_decoder):
    result_text = whisper_decoder.decode_audio("recorded_audio.wav")
    return result_text

async def send_data(websocket, path):
    recorder = AudioRecorder()
    whisper_decoder = SpeechToText()
    previous_emotion = None
    previous_text = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            try:
                await record_audio(recorder)

                emotion_future = asyncio.ensure_future(emotion_task())
                transcribe_future = asyncio.ensure_future(transcribe_task(whisper_decoder))

                await asyncio.gather(emotion_future, transcribe_future)

                major_emotion = emotion_future.result()
                text = transcribe_future.result()
                if major_emotion != previous_emotion or text != previous_text:
                    data = {'emotion': major_emotion, 'transcription': text}
                    print(data)
                    await websocket.send(json.dumps(data))

                    # Update previous data
                    previous_emotion = major_emotion
                    previous_text = text
            except KeyboardInterrupt:
                break


start_server = websockets.serve(send_data, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

