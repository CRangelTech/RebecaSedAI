import torch
import torch.nn as nn
import json
import sys
import os

# ---------- Rutas absolutas basadas en ubicación del archivo ----------
BASE_DIR = os.path.dirname(__file__)
PREPARED_PATH = os.path.join(BASE_DIR, "data", "prepared", "prepared_data.pt")
MODEL_PATH = os.path.join(BASE_DIR, "data", "trained", "uigen_model.pt")

# ---------- Clase del Modelo ----------
class UIModel(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.text_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.layout_embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_emb = self.text_embedding(x)
        _, (hidden, cell) = self.encoder(x_emb)

        batch_size = x.shape[0]
        max_len = 10

        input_token = torch.tensor([[SOS_IDX]] * batch_size)
        embedded = self.layout_embedding(input_token)
        hidden_state, cell_state = hidden, cell
        outputs = []

        for _ in range(max_len):
            out, (hidden_state, cell_state) = self.decoder(embedded, (hidden_state, cell_state))
            logits = self.fc(out[:, -1, :])
            predicted_idx = torch.argmax(logits, dim=1).item()

            if predicted_idx == EOS_IDX:
                break

            outputs.append(predicted_idx)
            embedded = self.layout_embedding(torch.tensor([[predicted_idx]]))

        return outputs

# ---------- Cargar modelo y vocabularios ----------
data = torch.load(PREPARED_PATH)
vocab = data["prompt_vocab"]
component_vocab = data["component_vocab"]
idx_to_component = {v: k for k, v in component_vocab.items()}

SOS_IDX = component_vocab["<SOS>"]
EOS_IDX = component_vocab["<EOS>"]

model = UIModel(
    input_dim=len(vocab),
    output_dim=len(component_vocab),
    emb_dim=32,
    hidden_dim=64
)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---------- Utilidades ----------
def encode_prompt(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]

def predict(prompt_text):
    encoded = encode_prompt(prompt_text)
    x = torch.tensor([encoded])
    predicted_indices = model(x)

    return [{"type": idx_to_component[i]} for i in predicted_indices]

# ---------- CLI / Server ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        layout = predict(prompt)
        print(json.dumps(layout, ensure_ascii=False))
    else:
        print("❌ Prompt vacío")
