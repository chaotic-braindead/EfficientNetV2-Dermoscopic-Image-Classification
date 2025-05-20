import os
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# === Load and Prepare Metadata ===
print("[INFO] Loading metadata...")
df = pd.read_csv("HAM10000_metadata.csv")  # Load metadata CSV

# Combine all images into one folder (only if needed)
os.makedirs("HAM10000_images", exist_ok=True)
for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
    for file in os.listdir(folder):
        shutil.copy(os.path.join(folder, file), "HAM10000_images")

# Add full image paths
df["path"] = df["image_id"].apply(lambda x: f"HAM10000_images/{x}.jpg")

# Encode the 'dx' label column (diagnosis) to numeric values
le = LabelEncoder()
df["label"] = le.fit_transform(df["dx"])
print("[INFO] Classes:", list(le.classes_))

# Split the dataset (80% train, 20% validation)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
print(f"[INFO] Train samples: {len(train_df)}, Val samples: {len(val_df)}")


# Custom Dataset class for PyTorch
class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")  # Open and convert image to RGB
        label = self.data.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)  # Apply image transforms
        return image, label


# Image transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet mean
        ),  # ImageNet std
    ]
)

# Create datasets and dataloaders
train_dataset = HAMDataset(train_df, transform=transform)
val_dataset = HAMDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("[INFO] DataLoaders created.")
# print(timm.list_models(pretrained=True))  # List available models
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. Install CUDA")
else:
    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    # Load pretrained EfficientNetV2 and change final layer to 7 classes
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 7)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train_model(model, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            print(f"\n[TRAIN] Epoch {epoch+1}/{epochs}")
            model.train()  # Set model to training mode
            total_loss, total_correct = 0, 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()

                if (i + 1) % 10 == 0:
                    print(f"[Batch {i+1}] Loss: {loss.item():.4f}")

            acc = total_correct / len(train_loader.dataset)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

            validate(model, val_loader)

    def validate(model, val_loader):
        model.eval()  # Set model to evaluation mode
        total_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                total_correct += (preds == labels).sum().item()

        acc = total_correct / len(val_loader.dataset)
        print(f"[VALIDATION] Accuracy: {acc:.4f}")

    train_model(model, train_loader, val_loader, epochs=10)

    def predict_image(model, image_path):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(1).item()
        print(f"[PREDICT] Raw output: {output.cpu().numpy()}")
        return le.inverse_transform([pred])[0]

    # Example usage
    result = predict_image(model, "HAM10000_images/ISIC_0027419.jpg")
    print(f"Predicted class: {result}")
