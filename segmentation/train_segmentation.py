# segmentation/train_segmentation.py

import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Set computation device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 1. Data Transformation
# ======================
transform = T.Compose([
    T.Resize((384, 384)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ======================
# 2. Load COCO Dataset
# ======================
dataset = CocoDetection(
    root='path_to/images/train2017',  # TODO: replace with actual image path
    annFile='path_to/annotations/instances_train2017.json',  # TODO: replace with actual annotation path
    transform=transform
)

# Use a collate function that unpacks image-target pairs
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ======================
# 3. Load Pretrained Mask R-CNN
# ======================
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE)

# ======================
# 4. Optimizer Setup
# ======================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

# ======================
# 5. Training Loop
# ======================
model.train()
losses_list = []

for epoch in range(10):
    epoch_loss = 0.0

    for images, targets in dataloader:
        # Move data to device
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    # Logging average loss for this epoch
    avg_loss = epoch_loss / len(dataloader)
    losses_list.append(avg_loss)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# ======================
# 6. Save Trained Model
# ======================
model_save_path = "segmentation/maskrcnn_finetuned.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved to {model_save_path}")

# ======================
# 7. Plot and Save Loss Curve
# ======================
plt.plot(losses_list, label="Segmentation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Segmentation Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("segmentation/loss_curve.png")
plt.show()
