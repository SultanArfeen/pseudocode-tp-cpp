import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import streamlit as st

# -------------------------------
# 0. File Paths & Checkpoint Path
# -------------------------------
BASE_DIR = "train"  # Adjust relative to your repository on Hugging Face Spaces
TRAIN_SPLIT = os.path.join(BASE_DIR, "split", "spoc-train-train.tsv")
EVAL_SPLIT  = os.path.join(BASE_DIR, "split", "spoc-train-eval.tsv")
FULL_DATA   = os.path.join(BASE_DIR, "spoc-train.tsv")  # optional
CKPT_PATH   = "q1_model.pth"

# -------------------------------
# 1. Dataset & Preprocessing
# -------------------------------
class SPoCDataset(Dataset):
    def __init__(self, tsv_file, mode='train'):
        self.mode = mode
        self.data = pd.read_csv(tsv_file, sep='\t', on_bad_lines='skip')
        grouped = self.data.groupby(['probid', 'subid'])
        self.samples = []
        for (_, _), group_df in grouped:
            group_df = group_df.sort_values(by='line')
            # Drop NaN and join lines
            pseudocode = "\n".join(group_df['text'].dropna().tolist())
            code       = "\n".join(group_df['code'].dropna().tolist())
            self.samples.append((pseudocode, code))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def simple_tokenizer(text):
    return text.lower().replace('\n', ' <newline> ').split()

class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.count = 4
    def add_sentence(self, tokens):
        for token in tokens:
            if token not in self.word2idx:
                self.word2idx[token] = self.count
                self.idx2word[self.count] = token
                self.count += 1
    def numericalize(self, tokens):
        return [self.word2idx.get(token, 3) for token in tokens]
    def denumericalize(self, indices):
        return [self.idx2word.get(i, "<unk>") for i in indices]

def build_vocab_spoc(dataset):
    src_vocab = Vocab()
    tgt_vocab = Vocab()
    for pseudocode, code in dataset:
        src_tokens = ["<sos>"] + simple_tokenizer(pseudocode) + ["<eos>"]
        tgt_tokens = ["<sos>"] + simple_tokenizer(code) + ["<eos>"]
        src_vocab.add_sentence(src_tokens)
        tgt_vocab.add_sentence(tgt_tokens)
    return src_vocab, tgt_vocab

# -------------------------------
# 2. Optimized Transformer Model
# -------------------------------
# We use smaller dimensions for faster training.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=64, nhead=2, 
                 num_encoder_layers=1, num_decoder_layers=1, 
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.src_embedding(src) * (self.d_model ** 0.5))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * (self.d_model ** 0.5))
        memory = self.transformer.encoder(src_emb)
        outputs = self.transformer.decoder(tgt_emb, memory)
        logits = self.fc_out(outputs)
        return logits

# -------------------------------
# 3. Training & Evaluation Utilities
# -------------------------------
def collate_fn_pseudocode_to_cpp(batch, src_vocab, tgt_vocab, max_len=128):
    src_seqs, tgt_seqs = [], []
    for pseudocode, code in batch:
        src_tokens = ["<sos>"] + simple_tokenizer(pseudocode) + ["<eos>"]
        tgt_tokens = ["<sos>"] + simple_tokenizer(code) + ["<eos>"]
        src_ids = src_vocab.numericalize(src_tokens)[:max_len]
        tgt_ids = tgt_vocab.numericalize(tgt_tokens)[:max_len]
        src_seqs.append(src_ids)
        tgt_seqs.append(tgt_ids)
    src_max = max(len(s) for s in src_seqs)
    tgt_max = max(len(s) for s in tgt_seqs)
    padded_src = [s + [0]*(src_max - len(s)) for s in src_seqs]
    padded_tgt = [t + [0]*(tgt_max - len(t)) for t in tgt_seqs]
    return torch.LongTensor(padded_src), torch.LongTensor(padded_tgt)

def train_step(model, src_batch, tgt_batch, criterion, optimizer, device):
    model.train()
    src_batch = src_batch.transpose(0, 1).to(device)
    tgt_batch = tgt_batch.transpose(0, 1).to(device)
    tgt_input = tgt_batch[:-1, :]
    tgt_target = tgt_batch[1:, :]
    optimizer.zero_grad()
    output = model(src_batch, tgt_input)
    output = output.reshape(-1, output.shape[-1])
    tgt_target = tgt_target.reshape(-1)
    loss = criterion(output, tgt_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_training_spoc(tsv_path, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Loading dataset: {os.path.basename(tsv_path)}")
    dataset = SPoCDataset(tsv_path, mode='train')
    st.info("Building vocabulary...")
    src_vocab, tgt_vocab = build_vocab_spoc(dataset)
    train_loader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=lambda batch: collate_fn_pseudocode_to_cpp(batch, src_vocab, tgt_vocab)
    )
    model = TransformerModel(
        src_vocab_size=len(src_vocab.word2idx),
        tgt_vocab_size=len(tgt_vocab.word2idx)
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_steps = epochs * len(train_loader)
    current_step = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state['cancel_training'] = False
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
            if st.session_state.get('cancel_training'):
                status_text.error("Training cancelled by user.")
                return None, None, None
            loss_val = train_step(model, src_batch, tgt_batch, criterion, optimizer, device)
            epoch_loss += loss_val
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        status_text.text(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    st.success("Training complete!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, CKPT_PATH)
    st.success(f"Model saved to {CKPT_PATH}")
    return model, src_vocab, tgt_vocab

def load_model_ckpt(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    model = TransformerModel(
        src_vocab_size=len(src_vocab.word2idx),
        tgt_vocab_size=len(tgt_vocab.word2idx)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, src_vocab, tgt_vocab

def translate_pseudocode_to_cpp(model, src_vocab, tgt_vocab, pseudocode, max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tokens = ["<sos>"] + simple_tokenizer(pseudocode) + ["<eos>"]
    src_ids = src_vocab.numericalize(tokens)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    tgt_indices = [tgt_vocab.word2idx["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = output[-1, 0, :].argmax().item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab.word2idx["<eos>"]:
            break
    result_tokens = tgt_vocab.denumericalize(tgt_indices)
    if result_tokens and result_tokens[0] == "<sos>":
        result_tokens = result_tokens[1:]
    if result_tokens and result_tokens[-1] == "<eos>":
        result_tokens = result_tokens[:-1]
    return " ".join(result_tokens)

# -------------------------------
# 5. Streamlit UI for Q1
# -------------------------------
def main():
    st.title("Pseudocode → C++ Converter")
    st.write("This Space trains a small Transformer model using the SPoC dataset to convert pseudocode into C++ code. Once trained, the model is saved for faster startup on subsequent runs.")

    st.sidebar.title("Actions")
    if st.sidebar.button("Load Saved Model (Q1)"):
        if os.path.exists(CKPT_PATH):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, src_vocab, tgt_vocab = load_model_ckpt(CKPT_PATH, device)
            st.session_state["model_q1"] = model
            st.session_state["src_vocab_q1"] = src_vocab
            st.session_state["tgt_vocab_q1"] = tgt_vocab
            st.sidebar.success("Loaded saved model.")
        else:
            st.sidebar.error("No saved model found. Please train first.")
    if st.sidebar.button("Train on Train Split (Q1)"):
        st.session_state['cancel_training'] = False
        out = run_training_spoc(TRAIN_SPLIT, epochs=2)
        if out[0] is not None:
            st.session_state["model_q1"], st.session_state["src_vocab_q1"], st.session_state["tgt_vocab_q1"] = out
    if st.sidebar.button("Train on Full Data (Q1)"):
        st.session_state['cancel_training'] = False
        out = run_training_spoc(FULL_DATA, epochs=2)
        if out[0] is not None:
            st.session_state["model_q1"], st.session_state["src_vocab_q1"], st.session_state["tgt_vocab_q1"] = out
    if st.sidebar.button("Cancel Training (Q1)"):
        st.session_state['cancel_training'] = True
        st.sidebar.info("Training cancellation requested.")

    st.write("---")
    st.header("Translate Pseudocode to C++")
    input_text = st.text_area("Enter pseudocode here:")
    if st.button("Translate", key="q1_translate"):
        if "model_q1" not in st.session_state:
            st.error("No model found. Please train or load a model first.")
        else:
            with st.spinner("Translating..."):
                output_cpp = translate_pseudocode_to_cpp(
                    st.session_state["model_q1"],
                    st.session_state["src_vocab_q1"],
                    st.session_state["tgt_vocab_q1"],
                    input_text
                )
                st.code(output_cpp, language='cpp')

if __name__ == "__main__":
    main()
