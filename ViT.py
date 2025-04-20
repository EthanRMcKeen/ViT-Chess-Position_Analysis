# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Constants
# ----------------------------
TRAIN_DATA_PATH = './dataset/train'
TEST_DATA_PATH = './dataset/test'
IMG_SIZE = 400
SQUARE_SIZE = IMG_SIZE // 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIECE_LABELS = list("prnbqkPRNBQK12345678")
NUM_CLASSES = 13
FEN_SYMBOLS = {'p':0,'r':1,'n':2,'b':3,'q':4,'k':5,'P':6,'R':7,'N':8,'B':9,'Q':10,'K':11,'1':12,'2':12,'3':12,'4':12,'5':12,'6':12,'7':12,'8':12}
MAX_IMAGES_TRAIN = 500
MAX_IMAGES_TEST = None

# ----------------------------
# Hyper-Parameters
# ----------------------------
PATCH_SIZE = 4
NUM_CHANNELS = 3
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 3
NUM_ATTN_HEADS = 1
INTERMEDIATE_SIZE = 256
HIDDEN_DROPOUT = 0.1
ATTN_DROPOUT = 0.1
LR = 1e-4


# ----------------------------
# Dataset
# ----------------------------
class ChessSquareDataset(Dataset):
    def __init__(self, data_path, transform=None, max_images=None):
        self.transform = transform
        self.samples = []
        files = [f for f in os.listdir(data_path) if f.endswith('.jpeg')]
        if max_images:
            files = files[:max_images]
        for fname in files:
            img_path = os.path.join(data_path, fname)
            fen = fname.split(".")[0].replace('-', '/')
            board = self.fen_to_labels(fen)
            self.samples.append((img_path, board))

    def fen_to_labels(self, fen):
        board = []
        for row in fen.split('/'):
            for char in row:
                if char.isdigit():
                    board.extend(['1'] * int(char))
                else:
                    board.append(char)
        return [FEN_SYMBOLS[c] for c in board]

    def __len__(self):
        return len(self.samples) * 64

    def __getitem__(self, idx):
        img_idx = idx // 64
        square_idx = idx % 64
        img_path, labels = self.samples[img_idx]
        img = Image.open(img_path).convert('RGB')
        row, col = divmod(square_idx, 8)
        square = img.crop((col*SQUARE_SIZE, row*SQUARE_SIZE, (col+1)*SQUARE_SIZE, (row+1)*SQUARE_SIZE))
        if self.transform:
            square = self.transform(square)
        return square, labels[square_idx]

# ----------------------------
# Model
# ----------------------------
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        config = ViTConfig(image_size=SQUARE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS,
                           num_labels=NUM_CLASSES, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS,
                           num_attention_heads=NUM_ATTN_HEADS, intermediate_size=INTERMEDIATE_SIZE,
                           output_attentions=True, hidden_dropout_prob=HIDDEN_DROPOUT,
                           attention_probs_dropout_prob=ATTN_DROPOUT)
        self.vit = ViTForImageClassification(config)

    def forward(self, x):
        return self.vit(x).logits

# ----------------------------
# Training
# ----------------------------
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

# ----------------------------
# Visualization
# ----------------------------
def visualize_attention(model, dataset):
    model.eval()
    img, label = dataset[1]  # second square of the first image
    with torch.no_grad():
        inputs = img.unsqueeze(0).to(DEVICE)
        outputs = model.vit(inputs, output_attentions=True)

        all_layer_attns = outputs.attentions  # list of [batch, heads, tokens, tokens]
        num_layers = len(all_layer_attns)
        num_heads = all_layer_attns[0].shape[1]

        # ensure axs is always 2D
        fig, axs = plt.subplots(num_layers, num_heads, figsize=(2 * num_heads, 2 * num_layers))
        if num_layers == 1 and num_heads == 1:
            axs = np.array([[axs]])
        elif num_layers == 1:
            axs = np.array([axs])
        elif num_heads == 1:
            axs = np.expand_dims(axs, axis=1)

        fig.suptitle("Attention Maps (CLS to Patches) — All Layers & Heads")

        for layer in range(num_layers):
            for head in range(num_heads):
                attn = all_layer_attns[layer][0, head, 0, 1:]  # CLS token's attention to patch tokens
                side = int(attn.shape[0] ** 0.5)
                attn_map = attn.reshape(side, side).cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

                axs[layer, head].imshow(attn_map, cmap='viridis')
                axs[layer, head].axis('off')
                axs[layer, head].set_title(f"L{layer+1}H{head+1}", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # grid of attention maps for first image
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for idx in range(64):
        square_img, _ = dataset[idx]  # first image's squares: idx 0-63
        with torch.no_grad():
            inputs = square_img.unsqueeze(0).to(DEVICE)
            outputs = model.vit(inputs, output_attentions=True)
            attn_weights = outputs.attentions[-1][0]
            avg_attn = attn_weights.mean(0)[0][1:]  # remove CLS
            side = int(len(avg_attn) ** 0.5)
            heatmap = avg_attn.reshape(side, side).cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            heatmap = Image.fromarray(np.uint8(heatmap * 255)).resize((SQUARE_SIZE, SQUARE_SIZE))

        axs[idx // 8, idx % 8].imshow(heatmap, cmap='jet')
        axs[idx // 8, idx % 8].axis('off')

    plt.tight_layout()
    plt.suptitle('Attention Maps of First Image Squares')
    plt.show()


def visualize_first_image_squares(dataset):
    img_path, _ = dataset.samples[0]
    img = Image.open(img_path).convert('RGB')
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Original Chessboard Image')
    plt.show()

    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for row in range(8):
        for col in range(8):
            square = img.crop((col*SQUARE_SIZE, row*SQUARE_SIZE, (col+1)*SQUARE_SIZE, (row+1)*SQUARE_SIZE))
            axs[row, col].imshow(square)
            axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()
# %%
# ----------------------------
# Main
# ----------------------------
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), # for greyscale images
    transforms.Resize((SQUARE_SIZE, SQUARE_SIZE)),
    transforms.ToTensor()
])

train_dataset = ChessSquareDataset(TRAIN_DATA_PATH, transform, max_images=MAX_IMAGES_TRAIN)
test_dataset = ChessSquareDataset(TEST_DATA_PATH, transform, max_images=MAX_IMAGES_TEST)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = VisionTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(10):
    loss = train(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

evaluate(model, test_loader)
visualize_attention(model, train_dataset)
visualize_first_image_squares(train_dataset)

# %%
# visulaize attention for a specific index

def visualize_attention_idx(model, dataset, idx):
    model.eval()
    img, label = dataset[idx] 
    with torch.no_grad():
        inputs = img.unsqueeze(0).to(DEVICE)
        outputs = model.vit(inputs, output_attentions=True)

        all_layer_attns = outputs.attentions  # list of [batch, heads, tokens, tokens]
        num_layers = len(all_layer_attns)
        num_heads = all_layer_attns[0].shape[1]

        # ensure axs is always 2D
        fig, axs = plt.subplots(num_layers, num_heads, figsize=(2 * num_heads, 2 * num_layers))
        if num_layers == 1 and num_heads == 1:
            axs = np.array([[axs]])
        elif num_layers == 1:
            axs = np.array([axs])
        elif num_heads == 1:
            axs = np.expand_dims(axs, axis=1)

        fig.suptitle("Attention Maps (CLS to Patches) — All Layers & Heads")
        for layer in range(num_layers):
            for head in range(num_heads):
                attn = all_layer_attns[layer][0, head, 0, 1:]  # CLS token's attention to patch tokens
                side = int(attn.shape[0] ** 0.5)
                attn_map = attn.reshape(side, side).cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

                axs[layer, head].imshow(attn_map, cmap='viridis')
                axs[layer, head].axis('off')
                axs[layer, head].set_title(f"L{layer+1}H{head+1}", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
visualize_attention_idx(model, train_dataset, 22)
# %%
def visualize_specific_square(dataset, row, col):
    img_path, _ = dataset.samples[0]
    img = Image.open(img_path).convert('RGB')
    square = img.crop((col * SQUARE_SIZE, row * SQUARE_SIZE, (col + 1) * SQUARE_SIZE, (row + 1) * SQUARE_SIZE))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(square)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


visualize_specific_square(train_dataset, 2, 6)
# %%
