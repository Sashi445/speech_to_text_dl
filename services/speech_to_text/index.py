from utils.speech_to_text.preprocess import mel_transform, greedy_decode
from models.speech_to_text.rnn_bi_lstm_ctc import SpeechRNNCTC
import torch
import torchaudio

class SpeechToTextService:
    def __init__(self):
        print("SpeechToTextService initialized")
        self.model_cache = {}

    def preprocess(self, audio_path, model_type="rnn_bi_lstm_ctc"):
        if model_type == "rnn_bi_lstm_ctc":
            try:
                waveform, sr = torchaudio.load(audio_path)
                mel = mel_transform(waveform).squeeze(0).transpose(0, 1)    
                mel = mel.unsqueeze(0)
                return mel
            except Exception as e:
                raise ValueError(f"Error preprocessing audio: {e}")
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def load_model(self, model_type="rnn_bi_lstm_ctc"):
        if model_type in self.model_cache:
            return self.model_cache[model_type]

        try:
            if model_type == "rnn_bi_lstm_ctc":
                model = SpeechRNNCTC()
                model.load_state_dict(torch.load("model_weights/speech_to_text_model.pth", map_location=torch.device("cpu")))
                model.eval()
                self.model_cache[model_type] = model
                return model
            else:
                raise ValueError(f"Model type {model_type} not supported")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def transcribe(self, audio_path, model_type="rnn_bi_lstm_ctc"):
        audio_features = self.preprocess(audio_path, model_type)
        model = self.load_model(model_type)
        with torch.no_grad():
            log_probs = model(audio_features)
            transcripts = greedy_decode(log_probs)
        return transcripts

    def transcribe_chunk(self, waveform, sample_rate=16000, model_type="rnn_bi_lstm_ctc"):
        try:
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            print("waveform shape before transform:", waveform.shape)

            mel = mel_transform(waveform)
            print("mel shape after mel_transform:", mel.shape)

            if mel.ndim == 3:
                mel = mel.squeeze(0)  # (mel, time)
            elif mel.ndim != 2:
                raise ValueError(f"Unexpected mel shape: {mel.shape}")

            mel = mel.transpose(0, 1).unsqueeze(0)  # (1, time, mel)
            print("mel shape before model:", mel.shape)

            model = self.load_model(model_type)

            with torch.no_grad():
                log_probs = model(mel)
                transcripts = greedy_decode(log_probs, cc=True)

            return transcripts

        except Exception as e:
            raise RuntimeError(f"Error in transcribing chunk: {e}")
