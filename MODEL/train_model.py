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
component_vocab = data["component_vocab"]
encoded_prompts = data["encoded_prompts"]
encoded_layouts = data["encoded_layouts"]

PAD_IDX = vocab["<PAD>"]
OUTPUT_PAD = component_vocab["<PAD>"]
SOS_IDX = component_vocab["<SOS>"]
EOS_IDX = component_vocab["<EOS>"]

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(component_vocab)

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
    y_batch = [pad_sequence(y, OUTPUT_PAD, max_y) for y in y_batch]
    return torch.tensor(x_batch), torch.tensor(y_batch)

# ---------- Modelo ----------
class UIModel(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.text_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.layout_embedding = nn.Embedding(output_dim, emb_dim, padding_idx=OUTPUT_PAD)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y=None, teacher_forcing=True):
        x_emb = self.text_embedding(x)
        _, (hidden, cell) = self.encoder(x_emb)

        if y is not None and teacher_forcing:
            y_emb = self.layout_embedding(y[:, :-1])
            outputs, _ = self.decoder(y_emb, (hidden, cell))
            logits = self.fc(outputs)
            return logits
        else:
            batch_size = x.shape[0]
            max_len = 10
            outputs = []
            input_token = torch.tensor([[SOS_IDX]] * batch_size).to(x.device)
            embedded = self.layout_embedding(input_token)
            hidden_state, cell_state = hidden, cell

            for _ in range(max_len):
                out, (hidden_state, cell_state) = self.decoder(embedded, (hidden_state, cell_state))
                logits = self.fc(out[:, -1, :])
                outputs.append(logits.unsqueeze(1))
                next_token = torch.argmax(logits, dim=1).unsqueeze(1)
                embedded = self.layout_embedding(next_token)

            return torch.cat(outputs, dim=1)

# ---------- Entrenamiento ----------
def train():
    dataset = UIDataset(encoded_prompts, encoded_layouts)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = UIModel(INPUT_DIM, OUTPUT_DIM, emb_dim=32, hidden_dim=64)
    criterion = nn.CrossEntropyLoss(ignore_index=OUTPUT_PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        total_loss = 0
        for x_batch, y_batch in loader:
            logits = model(x_batch, y_batch)
            targets = y_batch[:, 1:]
            loss = criterion(logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1))
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
