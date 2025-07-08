import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
import os

MAX_LEN = 64

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1, pad_idx_en=0, pad_idx_de=0):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_idx_en)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=pad_idx_de)
        self.pos_encoder = nn.Embedding(MAX_LEN + 2, emb_dim)
        self.transformer = nn.Transformer(
            d_model=emb_dim, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.generator = nn.Linear(emb_dim, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        src_seq_len, N = src.shape
        tgt_seq_len, N = tgt.shape
        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(1).expand(src_seq_len, N)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(1).expand(tgt_seq_len, N)
        src_emb = self.src_tok_emb(src) + self.pos_encoder(src_pos)
        tgt_emb = self.tgt_tok_emb(tgt) + self.pos_encoder(tgt_pos)
        src_padding_mask = (src == self.src_tok_emb.padding_idx).transpose(0, 1).bool()
        tgt_padding_mask = (tgt == self.tgt_tok_emb.padding_idx).transpose(0, 1).bool()
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.generator(out)

class TranslationPreprocessor:
    def __init__(self):
        # Always resolve paths relative to the root of your project
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tokenizer_dir = os.path.join(project_root, 'tokenizers')
        model_path = os.path.join(project_root, 'model_weights', 'best_transformer_model_bleu.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load English tokenizer
        self.en_tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "en_tokenizer-vocab.json"),
            os.path.join(tokenizer_dir, "en_tokenizer-merges.txt")
        )
        # Load German tokenizer
        self.de_tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "de_tokenizer-vocab.json"),
            os.path.join(tokenizer_dir, "de_tokenizer-merges.txt")
        )
        # Special token indices
        self.PAD_IDX_EN = self.en_tokenizer.token_to_id("<pad>")
        self.SOS_IDX_EN = self.en_tokenizer.token_to_id("<sos>")
        self.EOS_IDX_EN = self.en_tokenizer.token_to_id("<eos>")
        self.UNK_IDX_EN = self.en_tokenizer.token_to_id("<unk>")
        self.PAD_IDX_DE = self.de_tokenizer.token_to_id("<pad>")
        self.SOS_IDX_DE = self.de_tokenizer.token_to_id("<sos>")
        self.EOS_IDX_DE = self.de_tokenizer.token_to_id("<eos>")
        self.UNK_IDX_DE = self.de_tokenizer.token_to_id("<unk>")
        if any(idx is None for idx in [
            self.PAD_IDX_EN, self.SOS_IDX_EN, self.EOS_IDX_EN, self.UNK_IDX_EN,
            self.PAD_IDX_DE, self.SOS_IDX_DE, self.EOS_IDX_DE, self.UNK_IDX_DE
        ]):
            raise ValueError("Special tokens missing in tokenizer.")

        # Load Transformer model
        self.model = TransformerModel(
            len(self.en_tokenizer.get_vocab()), 
            len(self.de_tokenizer.get_vocab()),
            pad_idx_en=self.PAD_IDX_EN,
            pad_idx_de=self.PAD_IDX_DE
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _encode_bpe(self, text, tokenizer, sos_idx, eos_idx, max_len=MAX_LEN):
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        res_ids = [sos_idx] + ids + [eos_idx]
        if len(res_ids) > max_len:
            res_ids = res_ids[:max_len-1] + [eos_idx]
        return res_ids

    def translate(self, src_sentence, max_len=MAX_LEN):
        self.model.eval()
        src_ids = torch.tensor(self._encode_bpe(
            src_sentence, self.en_tokenizer, self.SOS_IDX_EN, self.EOS_IDX_EN
        ), dtype=torch.long).unsqueeze(1).to(self.device)
        src_len = src_ids.shape[0]
        src_pos = torch.arange(0, src_len, device=src_ids.device).unsqueeze(1).expand(src_len, 1)
        src_emb = self.model.src_tok_emb(src_ids) + self.model.pos_encoder(src_pos)
        src_padding_mask_single = (src_ids == self.PAD_IDX_EN).transpose(0, 1).bool()
        memory = self.model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask_single)
        ys = torch.tensor([[self.SOS_IDX_DE]], dtype=torch.long).to(self.device)
        for _ in range(max_len - 1):
            tgt_len = ys.shape[0]
            tgt_pos = torch.arange(0, tgt_len, device=ys.device).unsqueeze(1).expand(tgt_len, 1)
            tgt_emb = self.model.tgt_tok_emb(ys) + self.model.pos_encoder(tgt_pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            out = self.model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask_single
            )
            logits = self.model.generator(out[-1, :])
            next_word_id = logits.argmax(1).item()
            ys = torch.cat([ys, torch.tensor([[next_word_id]], device=self.device)], dim=0)
            if next_word_id == self.EOS_IDX_DE:
                break
        pred_tokens_ids = [id_val for id_val in ys[1:-1, 0].cpu().numpy()
                           if id_val not in [self.PAD_IDX_DE, self.SOS_IDX_DE, self.EOS_IDX_DE]]
        pred_sentence = self.de_tokenizer.decode(pred_tokens_ids) 
        return pred_sentence
