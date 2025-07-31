import torch
import torch.nn as nn
import os
import re

# ---------- Utilidades ----------
def clean_text(text):
    text = re.sub(r"[^\wáéíóúüñ]+", " ", text.lower())
    return text.strip()

def encode_prompt(text, vocab):
    cleaned = clean_text(text)
    return [vocab.get(word, vocab["<UNK>"]) for word in cleaned.split()]

# ---------- Modelo ----------
class UIModel(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, pad_idx, output_dim=5):
        super().__init__()
        self.text_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, max_len=20):
        x_emb = self.text_embedding(x)
        _, (hidden, cell) = self.encoder(x_emb)

        batch_size = x.size(0)
        device = x.device
        outputs = []

        input_token = torch.tensor([[type_vocab["<SOS>"]]], device=device)
        embedded = self.text_embedding(input_token)

        hidden_state, cell_state = hidden, cell

        for _ in range(max_len):
            out, (hidden_state, cell_state) = self.decoder(embedded, (hidden_state, cell_state))
            pred = self.output_layer(out[:, -1, :])  # pred.shape = (1, 5)
            outputs.append(pred[0])

            next_type_id = int(pred[0][0].round().clamp(min=0, max=len(type_vocab) - 1))
            if next_type_id == type_vocab["<EOS>"]:
                break

            embedded = self.text_embedding(torch.tensor([[next_type_id]], device=device))

        return outputs

# ---------- Cargar recursos ----------
PREPARED_PATH = os.path.join("data", "prepared", "prepared_data.pt")
MODEL_PATH = os.path.join("data", "trained", "uigen_model.pt")

data = torch.load(PREPARED_PATH)
vocab = data["prompt_vocab"]
type_vocab = data["type_vocab"]
PAD_IDX = vocab["<PAD>"]

idx_to_type = {v: k for k, v in type_vocab.items()}

# ---------- Cargar modelo ----------
INPUT_DIM = len(vocab)
model = UIModel(INPUT_DIM, emb_dim=32, hidden_dim=64, pad_idx=PAD_IDX)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------- Inferencia ----------
def predict(prompt_text):
    input_ids = encode_prompt(prompt_text, vocab)
    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(x)

    layout = []
    for vec in outputs:
        type_id = int(vec[0].round().clamp(min=0, max=len(type_vocab)-1))
        if type_id == type_vocab["<EOS>"]:
            break
        layout.append({
            "type": idx_to_type.get(type_id, "UNKNOWN"),
            "x": int(vec[1].item()),
            "y": int(vec[2].item()),
            "width": int(vec[3].item()),
            "height": int(vec[4].item()),
        })

    return layout

# ---------- CLI para probar ----------
if __name__ == "__main__":
    prompt = input("📝 Ingresa el prompt: ")
    layout = predict(prompt)
    print("\n📐 Layout generado:")
    for i, comp in enumerate(layout):
        print(f"{i+1:2d}. {comp}")
