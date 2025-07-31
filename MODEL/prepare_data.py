import json
from collections import Counter
import torch
import os
import re

# ---------- Rutas ----------
RAW_PATH = os.path.join("data", "unificado.json")
PREPARED_PATH = os.path.join("data", "prepared", "prepared_data.pt")

# ---------- Cargar el dataset ----------
with open(RAW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------- Procesar texto ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\wáéíóúüñ]+", " ", text)  # solo letras, números y tildes
    return text.strip()

all_texts = [clean_text(item["prompt"]).split() for item in data]
word_counts = Counter(word for sentence in all_texts for word in sentence)

vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common())}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_prompt(text):
    cleaned = clean_text(text)
    return [vocab.get(word, vocab["<UNK>"]) for word in cleaned.split()]

encoded_prompts = [encode_prompt(item["prompt"]) for item in data]

# ---------- Procesar layouts ----------
all_components = [comp["type"] for item in data for comp in item["layout"]]
component_counts = Counter(all_components)
component_vocab = {comp: idx for idx, (comp, _) in enumerate(component_counts.most_common())}
component_vocab["<PAD>"] = len(component_vocab)
component_vocab["<SOS>"] = len(component_vocab)
component_vocab["<EOS>"] = len(component_vocab)

def encode_layout(layout):
    return [component_vocab.get(comp["type"], component_vocab["<PAD>"]) for comp in layout]

encoded_layouts = [
    [component_vocab["<SOS>"]] + encode_layout(item["layout"]) + [component_vocab["<EOS>"]]
    for item in data
]

# ---------- Guardar datos ----------
os.makedirs(os.path.dirname(PREPARED_PATH), exist_ok=True)
torch.save({
    "prompt_vocab": vocab,
    "component_vocab": component_vocab,
    "encoded_prompts": encoded_prompts,
    "encoded_layouts": encoded_layouts
}, PREPARED_PATH)

print("✅ Datos preparados y guardados en:", PREPARED_PATH)
