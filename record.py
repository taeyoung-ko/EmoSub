import pyaudio
import numpy as np
import wave

class AudioRecorder:
    def __init__(self, output_filename="recorded_audio.wav", sample_rate=16000, chunk_size=1024, duration=5, threshold=0.1, input_device_index=None):
        self.output_filename = output_filename
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.duration = duration
        self.threshold = threshold
        self.input_device_index = input_device_index

    def get_available_microphones(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        devices = []

        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            devices.append((i, device_info['name']))

        p.terminate()
        return devices

    def record_audio(self):
        p = pyaudio.PyAudio()

        if self.input_device_index is None:
            # Choose the first available microphone by default
            devices = self.get_available_microphones()
            if devices:
                self.input_device_index = devices[0][0]
            else:
                raise ValueError("No microphone found.")

        stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index)

        print(f"Listening for sound on microphone {self.input_device_index}...")

        frames = []
        recording = False
        silent_frames = 0

        try:
            for i in range(0, int(self.sample_rate / self.chunk_size * self.duration)):
                data = stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.int16)

                if not recording and np.max(np.abs(audio_data)) > self.threshold * 32767:
                    print("Sound detected! Recording started.")
                    recording = True

                if recording:
                    frames.append(data)

                if recording and np.max(np.abs(audio_data)) < self.threshold * 32767:
                    silent_frames += 1
                    if silent_frames > 2 * self.sample_rate / self.chunk_size:
                        print("2 seconds of silence detected. Recording stopped.")
                        break
                else:
                    silent_frames = 0

        except KeyboardInterrupt:
            pass  

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        return frames   
    
    def filter_noise(self, frames):
        filtered_frames = []
        for frame in frames:
            audio_data = np.frombuffer(frame, dtype=np.int16)
            filtered_data = np.convolve(audio_data, np.ones(5)/5, mode='same') 
            filtered_frames.append(filtered_data.astype(np.int16).tobytes())
        return filtered_frames
    
    def save_audio(self, frames):
        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(1)  
        wf.setsampwidth(2) 
        wf.setframerate(self.sample_rate)  
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Audio saved to {self.output_filename}")

if __name__ == "__main__":
    recorder = AudioRecorder()
    print("Available microphones:")
    print(recorder.get_available_microphones())
    recorded_frames = recorder.record_audio()
    filtered_frames = recorder.filter_noise(recorded_frames)
    recorder.save_audio(filtered_frames)
