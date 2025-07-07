import torch
from torchaudio.transforms import MelSpectrogram

CHAR_VOCAB = ['<blank>'] + list("abcdefghijklmnopqrstuvwxyz '")
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

def text_to_indices(text):
    return [CHAR2IDX[c] for c in text.lower() if c in CHAR2IDX]

def greedy_decode(log_probs, cc=False):
    best_path = torch.argmax(log_probs, dim=-1)
    transcripts = []
    for seq in best_path:
        prev = None
        tokens = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                tokens.append(IDX2CHAR[idx])
            prev = idx
        transcripts.append("".join(tokens))

    if cc:
        return "".join(transcripts)

    return transcripts

mel_transform = MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
)