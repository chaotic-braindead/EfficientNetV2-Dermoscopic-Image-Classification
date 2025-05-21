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
import matplotlib.pyplot as plt
import numpy as np
import cv2

# === Load and Prepare Metadata ===
print("[INFO] Loading metadata...")
df = pd.read_csv("HAM10000_metadata.csv")

# Combine all images into one folder (only if needed)
if not os.path.exists("HAM10000_images"):
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

# First split: separate test set (80% train+val, 20% test)
train_val_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# Second split: separate train and validation from remaining data (80% train, 20% val)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.2, stratify=train_val_df["label"], random_state=42
)

print(f"[INFO] Train samples: {len(train_df)}")
print(f"[INFO] Validation samples: {len(val_df)}")
print(f"[INFO] Test samples: {len(test_df)}")

# Save split datasets for later use
train_df.to_csv("train_set.csv", index=False)
val_df.to_csv("val_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)


class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


# Image transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Create datasets and dataloaders
train_dataset = HAMDataset(train_df, transform=transform)
val_dataset = HAMDataset(val_df, transform=transform)
test_dataset = HAMDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("[INFO] DataLoaders created.")

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. Install CUDA")
else:
    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    model = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train_model(model, train_loader, val_loader, epochs=10):
        best_val_acc = 0.0
        for epoch in range(epochs):
            print(f"\n[TRAIN] Epoch {epoch+1}/{epochs}")
            model.train()
            total_loss, total_correct = 0, 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()

                if (i + 1) % 10 == 0:
                    print(f"[Batch {i+1}] Loss: {loss.item():.4f}")

            acc = total_correct / len(train_loader.dataset)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

            val_acc = validate(model, val_loader)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")

    def validate(model, val_loader):
        model.eval()
        total_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                total_correct += (preds == labels).sum().item()

        acc = total_correct / len(val_loader.dataset)
        print(f"[VALIDATION] Accuracy: {acc:.4f}")
        return acc

    def evaluate_test_set(model, test_loader):
        model.eval()
        total_correct = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)

                predictions.extend(preds.cpu().numpy())
                actuals.extend(labels.cpu().numpy())
                total_correct += (preds == labels).sum().item()

        acc = total_correct / len(test_loader.dataset)
        print(f"[TEST] Final Accuracy: {acc:.4f}")

        # Convert numeric predictions back to class names
        pred_classes = le.inverse_transform(predictions)
        actual_classes = le.inverse_transform(actuals)

        return pred_classes, actual_classes, acc

    def predict_and_display(model, image_id, dataset="test"):
        """
        Predict and display results for an image from validation or test set

        Args:
            model: The trained model
            image_id: The ISIC_* id of the image
            dataset: Either 'test' or 'val' to specify which set to use
        """
        # Verify dataset choice
        if dataset not in ["test", "val"]:
            raise ValueError("dataset must be either 'test' or 'val'")

        # Get the appropriate dataframe
        df_to_use = test_df if dataset == "test" else val_df

        # Check if image_id exists in the specified dataset
        if not df_to_use["image_id"].str.contains(image_id).any():
            raise ValueError(f"Image {image_id} not found in the {dataset} dataset")

        # Get image path
        image_path = f"HAM10000_images/{image_id}.jpg"

        # Original prediction code
        model.eval()
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(1).item()

        predicted_class = le.inverse_transform([pred])[0]
        actual_class = df_to_use[df_to_use["image_id"] == image_id]["dx"].values[0]

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Predicted: {predicted_class}\nActual: {actual_class}\nSet: {dataset}",
            pad=20,
        )
        plt.show()

        return predicted_class, actual_class

    def generate_gradcam(model, image_path, class_idx=None):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Hook into last convolutional layer
        target_layer = model.blocks[-1][-1]
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item() if class_idx is None else class_idx
        df_to_use = test_df
        actual_class = df_to_use[df_to_use["image_id"] == image_id]["dx"].values[0]
        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0].detach().cpu()
        acts = activations[0].detach().cpu()
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        for i in range(acts.shape[1]):
            acts[0, i, :, :] *= pooled_grads[i]
        heatmap = acts[0].sum(dim=0)
        heatmap = torch.relu(heatmap)
        heatmap /= heatmap.max()

        heatmap = transforms.ToPILImage()(heatmap.unsqueeze(0)).resize(
            image.size, Image.Resampling.BILINEAR
        )
        heatmap = np.array(heatmap)
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)

        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(
            f"Predicted: {le.inverse_transform([pred_class])[0]}\nActual: {actual_class}\nImage ID: {image_path.split('/')[-1]}"
        )
        plt.show()

        forward_handle.remove()
        backward_handle.remove()

    # Function to get random samples from test/val sets
    def get_random_samples(dataset="test", n_samples=5):
        """Get random sample image IDs from test or validation set"""
        df_to_use = test_df if dataset == "test" else val_df
        return df_to_use["image_id"].sample(n=n_samples).tolist()

    # Train the model if not already trained
    if os.path.exists("best_model.pth"):
        resp = input("retrain model? (y/n): ")
        if resp.lower() != "y":
            print("[INFO] Skipping training.")
        else:
            epochs = int(input("Enter number of epochs: "))
            train_model(model, train_loader, val_loader, epochs=epochs)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("best_model.pth"))
    pred_classes, actual_classes, test_acc = evaluate_test_set(model, test_loader)

    # # Example prediction
    # result_pred, result_actual = predict_and_display(
    #     model, "HAM10000_images/ISIC_0030555.jpg"
    # )

    for image_id in get_random_samples(dataset="test", n_samples=len(test_df)):
        predict_and_display(model, image_id, dataset="test")
        generate_gradcam(model, f"HAM10000_images/{image_id}.jpg")

    # print(f"\nPredicted class: {result_pred}")
    # print(f"Actual class: {result_actual}")
