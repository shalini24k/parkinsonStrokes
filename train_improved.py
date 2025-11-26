import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# --- CONFIG ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8          # MORE epochs for better learning
LR = 0.0001         # LOWER learning rate for precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODEL ---
class PDDenseNetImproved(nn.Module):
    def __init__(self, num_classes=2):
        super(PDDenseNetImproved, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.features = self.densenet.features
        in_features = self.densenet.classifier.in_features
        # IMPROVED Head (Dropout + Linear)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Dropout reduces overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def train_routine(train_dir, save_name):
    print(f"\n--- Training Improved Model: {save_name} ---")
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Folder {train_dir} not found!")
        return

    # IMPROVED Transform (Data Augmentation)
    # This creates "fake" new images by rotating/flipping, making the model smarter
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data = ImageFolder(train_dir, transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PDDenseNetImproved(num_classes=len(data.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1) # Smart learning rate
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
        
        scheduler.step() # Update learning rate
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), save_name)
    print(f"✅ Saved: {save_name}")

if __name__ == "__main__":
    train_routine("./dataset_strokes/train", "model_strokes_improved.pth")
    train_routine("./dataset_words/dataset-word/Train", "model_words_improved.pth")
