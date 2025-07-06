from utils.speech_to_text.preprocess import mel_transform, greedy_decode
from models.speech_to_text.rnn_bi_lstm_ctc import SpeechRNNCTC
import torch
import torchaudio

class SpeechToTextService:
    def __init__(self):
        print("SpeechToTextService initialized")
    
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
        try:
            if model_type == "rnn_bi_lstm_ctc":
                model = SpeechRNNCTC()
                model.load_state_dict(torch.load("model_weights/speech_to_text_model.pth", map_location=torch.device("cpu")))
                model.eval()
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
            if model_type == "rnn_bi_lstm_ctc": 
                transcripts = greedy_decode(log_probs)
            else:
                raise ValueError(f"Model type {model_type} not supported")
        return transcripts