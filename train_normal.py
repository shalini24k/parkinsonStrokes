import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# --- CONFIG ---
# We use 128x128 for speed. 
# If you have a GPU, you can change this to 224.
IMG_SIZE = 128 
BATCH_SIZE = 32
EPOCHS = 3          # Low epochs for "Normal" model
LR = 0.001          # High learning rate (less precise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODEL ---
class PDDenseNetNormal(nn.Module):
    def __init__(self, num_classes=2):
        super(PDDenseNetNormal, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.features = self.densenet.features
        in_features = self.densenet.classifier.in_features
        # Standard Linear Head (No Dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def train_routine(train_dir, save_name):
    print(f"\n--- Training Standard Model: {save_name} ---")
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Folder {train_dir} not found!")
        return

    # BASIC Transform (No Augmentation)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data = ImageFolder(train_dir, transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PDDenseNetNormal(num_classes=len(data.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), save_name)
    print(f"✅ Saved: {save_name}")

if __name__ == "__main__":
    # Train both datasets
    train_routine("./dataset_strokes/train", "model_strokes_normal.pth")
    train_routine("./dataset_words/dataset-word/train", "model_words_normal.pth")