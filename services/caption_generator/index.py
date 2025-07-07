import os
from services.speech_to_text.index import SpeechToTextService
from utils.video import extract_audio, chunk_audio, burn_captions

class CaptionGeneratorService:
    def __init__(self, model_type="rnn_bi_lstm_ctc"):
        self.stt_service = SpeechToTextService()
        self.model_type = model_type

    def process_video(self, video_path: str, output_path: str) -> str:
        """
        Main method to generate captioned video.
        :param video_path: Path to input video
        :param output_path: Path to save captioned video
        :return: Path to the captioned output video
        """
        audio_path = self.extract_audio(video_path)
        audio_chunks = self.chunk_audio(audio_path)
        captions = self.transcribe_chunks(audio_chunks)
        self.burn_captions(video_path, captions, output_path)
        return output_path

    def extract_audio(self, video_path: str) -> str:
        return extract_audio(video_path)

    def chunk_audio(self, audio_path: str, chunk_length: float = 5.0):
        return chunk_audio(audio_path, chunk_length)

    def transcribe_chunks(self, audio_chunks):
        """
        Transcribes audio chunks using the speech-to-text model.
        :param audio_chunks: List of (waveform, start, end)
        :return: List of {start, end, text}
        """
        captions = []
        for waveform, start, end in audio_chunks:
            text = self.stt_service.transcribe_chunk(waveform, sample_rate=16000, model_type=self.model_type)
            if text.strip():
                captions.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
        return captions

    def burn_captions(self, video_path: str, captions: list, output_path: str):
        burn_captions(video_path, captions, output_path)
