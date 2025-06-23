import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor
from tqdm import tqdm

# ========== Step 1: Device Setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ========== Step 2: Dataset ==========
class Flickr8kEncodedFeatureDataset(Dataset):
    def __init__(self, captions_file, image_list_file, encoded_pkl_file, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load valid image names
        with open(image_list_file, 'r') as f:
            valid_images = set(line.strip() for line in f)

        # Load encoded image features (2048-dim vectors)
        with open(encoded_pkl_file, 'rb') as f:
            encoded_np = pickle.load(f)
        self.encoded_images = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in encoded_np.items()
        }

        # Load captions
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                img_name_full, caption = parts
                img_name = img_name_full.split('#')[0]
                if img_name in valid_images and img_name in self.encoded_images:
                    self.data.append((img_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image_feat = self.encoded_images[img_name]  # Shape: [2048]

        encoding = self.tokenizer(caption, return_tensors="pt", padding='max_length',
                                  truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "image_feat": image_feat,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# ========== Step 3: Model ==========
class CaptioningModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, max_length=32):
        super().__init__()
        self.img_proj = nn.Linear(feature_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(self, image_feat, input_ids):
        batch_size = image_feat.size(0)
        img_emb = self.img_proj(image_feat).unsqueeze(1)  # [B, 1, H]
        text_emb = self.embedding(input_ids[:, :-1])      # [B, L-1, H]
        seq_input = torch.cat([img_emb, text_emb], dim=1) # [B, L, H]

        out, _ = self.lstm(seq_input)
        logits = self.fc(out)  # [B, L, vocab_size]
        return logits

# ========== Step 4: Paths ==========
CAPTIONS_FILE = "dataset/Flickr8k.token.txt"
IMAGE_LIST_FILE = "dataset/Flickr_8k.trainImages.txt"
ENCODED_IMAGE_PKL = "dataset/Pickle/encoded_train_images.pkl"

# ========== Step 5: Tokenizer & Dataset ==========
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size

dataset = Flickr8kEncodedFeatureDataset(
    captions_file=CAPTIONS_FILE,
    image_list_file=IMAGE_LIST_FILE,
    encoded_pkl_file=ENCODED_IMAGE_PKL,
    tokenizer=tokenizer,
    max_length=32
)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== Step 6: Model & Optimizer ==========
model = CaptioningModel(feature_dim=2048, hidden_dim=512, vocab_size=vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ========== Step 7: Training Loop ==========
print("Training started...")
EPOCHS = 20
losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        image_feat = batch["image_feat"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)

        logits = model(image_feat, input_ids)

        # Prepare target: next token in sequence
        targets = input_ids[:, 1:]  # Shifted right
        logits = logits[:, 1:, :].reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg = total_loss / len(train_loader)
    losses.append(avg)
    print(f"Epoch {epoch+1} completed. Avg loss: {avg:.4f}")

# ========== Step 8: Save Model ==========
os.makedirs("captioning/feature_model", exist_ok=True)
torch.save(model.state_dict(), "captioning/feature_model/model.pt")
print("Model saved to captioning/feature_model/model.pt")

# ========== Step 9: Plot Loss ==========
plt.plot(losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("captioning/feature_model/loss_plot.png")
plt.show()
