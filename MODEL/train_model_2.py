import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# ---------- Rutas ----------
PREPARED_PATH = os.path.join("data", "prepared", "prepared_data.pt")
TRAINED_MODEL_PATH = os.path.join("data", "trained", "uigen_model.pt")

# ---------- Cargar datos ----------
data = torch.load(PREPARED_PATH)
vocab = data["prompt_vocab"]
type_vocab = data["type_vocab"]
encoded_prompts = data["encoded_prompts"]
encoded_layouts = data["encoded_layouts"]

PAD_IDX = vocab["<PAD>"]
SOS_VEC = [type_vocab["<SOS>"], 0, 0, 0, 0]
EOS_VEC = [type_vocab["<EOS>"], 0, 0, 0, 0]

INPUT_DIM = len(vocab)

# ---------- Dataset ----------
class UIDataset(Dataset):
    def __init__(self, prompts, layouts):
        self.prompts = prompts
        self.layouts = layouts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.layouts[idx]

def pad_sequence(seq, pad_value, max_len):
    return seq + [pad_value] * (max_len - len(seq))

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    max_x = max(len(x) for x in x_batch)
    max_y = max(len(y) for y in y_batch)
    x_batch = [pad_sequence(x, PAD_IDX, max_x) for x in x_batch]
    y_batch = [pad_sequence(y, EOS_VEC, max_y) for y in y_batch]  # usa vectores [type_id, x, y, w, h]
    x_batch = torch.tensor(x_batch, dtype=torch.long)
    y_batch = torch.tensor(y_batch, dtype=torch.float)
    return x_batch, y_batch

# ---------- Modelo ----------
class UIModel(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.text_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 5)  # [type_id, x, y, w, h]

    def forward(self, x, y=None, teacher_forcing=True):
        x_emb = self.text_embedding(x)
        _, (hidden, cell) = self.encoder(x_emb)

        batch_size = x.shape[0]
        max_len = y.shape[1] if (y is not None and teacher_forcing) else 20
        device = x.device

        outputs = []
        input_token = torch.tensor([[type_vocab["<SOS>"]]] * batch_size, device=device)

        # Primera entrada: solo el type_id embebido (ignoramos xywh en decoder)
        embedded = self.text_embedding(input_token)

        hidden_state, cell_state = hidden, cell

        for t in range(max_len):
            out, (hidden_state, cell_state) = self.decoder(embedded, (hidden_state, cell_state))
            pred = self.output_layer(out[:, -1, :])  # pred.shape = (batch, 5)
            outputs.append(pred.unsqueeze(1))

            if teacher_forcing and y is not None:
                next_input_ids = y[:, t, 0].long()  # usar type_id real del siguiente paso
            else:
                next_input_ids = pred[:, 0].round().long().clamp(min=0, max=len(type_vocab) - 1)

            embedded = self.text_embedding(next_input_ids.unsqueeze(1))

        return torch.cat(outputs, dim=1)

# ---------- Entrenamiento ----------
def train():
    dataset = UIDataset(encoded_prompts, encoded_layouts)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = UIModel(INPUT_DIM, emb_dim=32, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        total_loss = 0
        for x_batch, y_batch in loader:
            preds = model(x_batch, y_batch)
            targets = y_batch[:, 1:]  # salta el <SOS>
            preds = preds[:, :targets.size(1), :]  # igualar longitudes
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📚 Epoch {epoch+1}/50 - Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(TRAINED_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print("✅ Modelo guardado en:", TRAINED_MODEL_PATH)

if __name__ == "__main__":
    train()
