#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

import numpy as np
import pyaudio
import wave
import librosa
import scipy
from scipy.stats import zscore

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM

import asyncio
import websockets
import json

class speechEmotionRecognition:

    def __init__(self, subdir_model=None):
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)

        self._emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    def voice_recording(self, filename, duration=1, sample_rate=16000, chunk=1024, channels=1):

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()

        while (current_time - start_time) < duration:

            data = stream.read(chunk)
            frames.append(data)
            current_time = time.time()

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording * ')

        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
        mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return np.asarray(mel_spect)

    def frame(self, y, win_step=64, win_size=128):
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
        for t in range(nb_frames):
            frames[:,t,:,:] = np.copy(y[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float16)

        return frames

    def build_model(self):
        K.clear_session()
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)
        y = Dense(7, activation='softmax', name='FC')(y)
        model = Model(inputs=input_y, outputs=y)

        return model

    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
        chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
        y = np.asarray(list(map(zscore, chunks)))
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))
        mel_spect_ts = self.frame(mel_spect)

        X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                    mel_spect_ts.shape[1],
                                    mel_spect_ts.shape[2],
                                    mel_spect_ts.shape[3],
                                    1)

        if predict_proba is True:
            predict = self._model.predict(X)
        else:
            predict = np.argmax(self._model.predict(X), axis=1)
            predict = [self._emotion.get(emotion) for emotion in predict]

        K.clear_session()

        timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
        timestamp = np.round(timestamp / sample_rate)

        return [predict, timestamp]
        #return predict
        
    def prediction_to_csv(self, predictions, filename, mode='w'):
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS"+'\n')
            for emotion in predictions:
                f.write(str(emotion)+'\n')
            f.close()

    def emotion_task(file_dir, model_sub_dir, SER, step, sample_rate):
        try:
            emotions, timestamp = SER.predict_emotion_from_file(file_dir, chunk_step=step*sample_rate)
            emotions = SER.predict_emotion_from_file(file_dir, chunk_step=step*sample_rate)
            major_emotion = max(set(emotions), key=emotions.count)
            print("timestamp: " + str(timestamp))
        except:
            major_emotion = "Not Detected"

        return major_emotion

def run_model(data_queue,phrase_time,last_sample,transcription,emotion,audio_filename,source,audio_model):

    now = datetime.utcnow()
    if not data_queue.empty():
        phrase_complete = False

        if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
            last_sample = bytes()
            phrase_complete = True
        phrase_time = now

        while not data_queue.empty():
            data = data_queue.get()
            last_sample += data

        audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        wav_data = io.BytesIO(audio_data.get_wav_data())

        with open(audio_filename, 'wb') as f:
            f.write(wav_data.read())

        result = audio_model.transcribe(audio_filename, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        text = text.replace(".", "")
        major_emotion = speechEmotionRecognition.emotion_task(audio_filename, 'audio.hdf5', speechEmotionRecognition('audio.hdf5'), 1, 16000)
        #major_emotion = "Not Detected"
        if phrase_complete:
            transcription.append(text)
            emotion.append(major_emotion)
        else:
            emotion[-1] = major_emotion
            transcription[-1] = text

        os.system('cls' if os.name=='nt' else 'clear')
        for i in range(len(transcription)):
            print(transcription[i] + " + " + emotion[i])
        print('', end='', flush=True)
        #sleep(0.25)
    return transcription, emotion

def setup():
    audio_filename = "recorded_audio.wav"
    #audio_filename = "test.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    args = parser.parse_args()

    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']
    emotion = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    return data_queue,phrase_time,last_sample,transcription,emotion,audio_filename,source,audio_model

async def echo(websocket, path):
    data_queue,phrase_time,last_sample,transcription,emotion,audio_filename,source,audio_model = setup()
    while True:
        try:
            t, e = run_model(data_queue,phrase_time,last_sample,transcription,emotion,audio_filename,source,audio_model)
            data = {'emotion': e[-1], 'transcription': t[-1]}
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break

start_server = websockets.serve(echo, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
