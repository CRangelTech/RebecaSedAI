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
    text = re.sub(r"[^\wáéíóúüñ]+", " ", text.lower())
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
all_types = [comp["type"] for item in data for comp in item["layout"]]
type_counts = Counter(all_types)
type_vocab = {t: idx for idx, (t, _) in enumerate(type_counts.most_common())}

# Reservar índices especiales para <PAD>, <SOS> y <EOS>
PAD_IDX = len(type_vocab)
SOS_IDX = PAD_IDX + 1
EOS_IDX = PAD_IDX + 2

type_vocab["<PAD>"] = PAD_IDX
type_vocab["<SOS>"] = SOS_IDX
type_vocab["<EOS>"] = EOS_IDX

def encode_component(comp):
    type_id = type_vocab.get(comp["type"], type_vocab["<PAD>"])
    x = int(comp.get("x", 0))
    y = int(comp.get("y", 0))
    width = int(comp.get("width", 0))
    height = int(comp.get("height", 0))
    return [type_id, x, y, width, height]

def encode_layout(layout):
    return [encode_component(comp) for comp in layout]

encoded_layouts = [
    [[SOS_IDX, 0, 0, 0, 0]] + encode_layout(item["layout"]) + [[EOS_IDX, 0, 0, 0, 0]]
    for item in data
]

# ---------- Guardar datos ----------
os.makedirs(os.path.dirname(PREPARED_PATH), exist_ok=True)
torch.save({
    "prompt_vocab": vocab,
    "type_vocab": type_vocab,
    "encoded_prompts": encoded_prompts,
    "encoded_layouts": encoded_layouts
}, PREPARED_PATH)

print("✅ Datos preparados y guardados en:", PREPARED_PATH)
