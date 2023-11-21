import whisper

class SpeechToText:
    def __init__(self, model_path="medium"):
        self.model = whisper.load_model(model_path)

    def load_audio(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        return audio

    def log_mel_spectrogram(self, audio):
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel

    def decode_audio(self, audio_path):
        audio = self.load_audio(audio_path)
        mel = self.log_mel_spectrogram(audio)

        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        return result.text


if __name__ == "__main__":
    whisper_decoder = SpeechToText()
    while True:
        result_text = whisper_decoder.decode_audio("test.wav")
        print(result_text)
