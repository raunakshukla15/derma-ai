import torch, json, os
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ================= DATA =================
train_ds = datasets.ImageFolder("dataset/train", transform=train_tf)
val_ds   = datasets.ImageFolder("dataset/val", transform=val_tf)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=16)

# ================= CLASS WEIGHTS =================
class_counts = np.bincount(train_ds.targets)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)

# ================= MODEL =================
model = models.efficientnet_b0(pretrained=True)

# ---- PHASE 1: FREEZE BACKBONE ----
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(train_ds.classes)
)

model.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1   # 🔑 critical for eczema vs acne
)

optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=3e-4
)

# ================= TRAINING =================
best_val_acc = 0
epochs_phase1 = 15
epochs_phase2 = 15

def validate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_dl:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ---- PHASE 1 TRAIN ----
for epoch in range(epochs_phase1):
    model.train()
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    val_acc = validate()
    print(f"[Phase 1] Epoch {epoch+1}/{epochs_phase1} | Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/model.pth")

# ---- PHASE 2: UNFREEZE LAST BLOCKS ----
for param in model.features[-2:].parameters():
    param.requires_grad = True

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# ---- PHASE 2 TRAIN ----
for epoch in range(epochs_phase2):
    model.train()
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    val_acc = validate()
    print(f"[Phase 2] Epoch {epoch+1}/{epochs_phase2} | Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/model.pth")

# ================= SAVE CLASSES =================
with open("model/classes.json","w") as f:
    json.dump(train_ds.classes, f)

print("✅ Training finished. Best model saved.")
