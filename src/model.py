"""
model.py
=========
Hybrid deep learning models for wildfire risk prediction.

Models implemented:
1. SpatialCNN - Baseline CNN for single-timestep spatial features
2. CNN_LSTM - Hybrid model: CNN extracts spatial features, LSTM captures temporal trends
3. WildfireTransformer - Vision Transformer variant with temporal attention

The model takes:
- Satellite imagery tiles (multi-band spatial features)
- Weather time-series (temperature, humidity, wind, etc.)
And predicts fire risk class: 0=No Risk, 1=Moderate, 2=High Risk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path


# ─────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────

class WildfireDataset(Dataset):
    """
    Dataset for wildfire prediction.
    
    Loads preprocessed tiles and optional weather time-series.
    Supports data augmentation (random flips, rotations).
    """
    
    def __init__(self, X_path, y_path, weather_path=None, augment=False):
        """
        Args:
            X_path: path to satellite tile numpy file (N, C, H, W)
            y_path: path to labels numpy file (N,)
            weather_path: optional path to weather time-series (N, T, W_features)
            augment: bool, whether to apply data augmentation
        """
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.weather = np.load(weather_path) if weather_path else None
        self.augment = augment
        
        print(f"  Dataset: {len(self.y)} samples, {self.X.shape[1]} channels, "
              f"{self.X.shape[2]}x{self.X.shape[3]} pixels")
        print(f"  Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])[0]
        
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-1])
            # Random vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-2])
            # Random 90-degree rotation
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [-2, -1])
        
        if self.weather is not None:
            w = torch.FloatTensor(self.weather[idx])
            return x, w, y
        
        return x, y


# ─────────────────────────────────────────────────
# MODEL 1: Spatial CNN (Baseline)
# ─────────────────────────────────────────────────

class SpatialCNN(nn.Module):
    """
    Baseline CNN for fire risk classification from a single satellite tile.
    
    Architecture:
    - 4 conv blocks with batch norm and max pooling
    - Global average pooling
    - 2 FC layers → fire risk class
    
    Good baseline to beat with the temporal models.
    """
    
    def __init__(self, in_channels, n_classes=2, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: (C, 64, 64) → (32, 32, 32)
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: (128, 8, 8) → (256, 4, 4)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (256, 1, 1)
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        """Extract spatial features (for use in CNN-LSTM)."""
        features = self.features(x)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)  # (batch, 256)


# ─────────────────────────────────────────────────
# MODEL 2: CNN-LSTM Hybrid
# ─────────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM for spatiotemporal wildfire prediction.
    
    How it works:
    1. CNN encoder extracts spatial features from each satellite tile
       in a temporal sequence (e.g., monthly composites over 6 months)
    2. LSTM processes the sequence of spatial feature vectors to
       capture temporal trends (drying vegetation, changing weather)
    3. Weather features are concatenated with CNN features at each timestep
    4. Final classifier predicts fire risk
    
    Input:
        satellite_seq: (batch, timesteps, channels, H, W) - satellite tile sequence
        weather_seq: (batch, timesteps, weather_features) - weather time series
    """
    
    def __init__(self, in_channels, weather_features=0, n_classes=2,
                 cnn_features=256, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        
        # CNN encoder (shared weights across timesteps)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, cnn_features, 3, padding=1),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (batch, cnn_features, 1, 1)
        )
        
        # LSTM for temporal modeling
        lstm_input_size = cnn_features + weather_features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, satellite_seq, weather_seq=None):
        """
        Args:
            satellite_seq: (batch, T, C, H, W)
            weather_seq: (batch, T, W) or None
        """
        batch_size, T, C, H, W = satellite_seq.shape
        
        # Process each timestep through CNN
        # Reshape to (batch*T, C, H, W) for batch processing
        sat_flat = satellite_seq.reshape(batch_size * T, C, H, W)
        cnn_features = self.cnn(sat_flat)  # (batch*T, cnn_features, 1, 1)
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)  # (batch*T, cnn_features)
        cnn_features = cnn_features.reshape(batch_size, T, -1)  # (batch, T, cnn_features)
        
        # Concatenate weather features if provided
        if weather_seq is not None:
            lstm_input = torch.cat([cnn_features, weather_seq], dim=-1)
        else:
            lstm_input = cnn_features
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden*2)
        
        return self.classifier(last_output)


# ─────────────────────────────────────────────────
# MODEL 3: Lightweight Vision Transformer
# ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, in_channels, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch, C, H, W) → (batch, embed_dim, H/P, W/P)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class WildfireTransformer(nn.Module):
    """
    Vision Transformer for wildfire prediction.
    
    Splits tiles into patches, adds positional encoding,
    processes through transformer encoder layers, and classifies.
    Can also incorporate temporal attention across timesteps.
    """
    
    def __init__(self, in_channels, img_size=64, patch_size=8,
                 embed_dim=128, n_heads=4, n_layers=4,
                 n_classes=2, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        
        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer
        x = self.transformer(x)
        
        # Classify from CLS token
        return self.head(x[:, 0])


# ─────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance.
    Wildfires are rare → most tiles are "no fire" → class imbalance.
    Focal loss down-weights easy examples, focuses on hard ones.
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_class_weights(labels):
    """Compute inverse frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.FloatTensor(weights)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        if len(batch) == 3:
            x, weather, y = batch
            x, weather, y = x.to(device), weather.to(device), y.to(device)
            logits = model(x, weather)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        if len(batch) == 3:
            x, weather, y = batch
            x, weather, y = x.to(device), weather.to(device), y.to(device)
            logits = model(x, weather)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        
        probs = F.softmax(logits, dim=1)
        _, predicted = logits.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    
    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, 
                                target_names=["No Fire", "Fire"],
                                zero_division=0))
    
    # AUC-ROC if binary
    if len(np.unique(all_labels)) == 2:
        probs_arr = np.array(all_probs)
        auc = roc_auc_score(all_labels, probs_arr[:, 1])
        print(f"  AUC-ROC: {auc:.4f}")
    
    return avg_loss, accuracy


# ─────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────

def train_model(data_dir="data/tiles", model_type="cnn", epochs=50, 
                batch_size=32, lr=1e-3, device=None):
    """
    Train a wildfire prediction model.
    
    Args:
        data_dir: directory with X_train.npy, y_train.npy, etc.
        model_type: 'cnn', 'cnn_lstm', or 'transformer'
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        device: 'cuda' or 'cpu'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    data = Path(data_dir)
    
    # Load metadata
    with open(data / "metadata.json") as f:
        meta = json.load(f)
    
    in_channels = meta["n_channels"]
    
    # Create datasets
    print("\nLoading datasets...")
    train_ds = WildfireDataset(
        data / "X_train.npy", data / "y_train.npy", augment=True
    )
    test_ds = WildfireDataset(
        data / "X_test.npy", data / "y_test.npy", augment=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_ds.y).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Create model
    print(f"\nModel: {model_type}")
    if model_type == "cnn":
        model = SpatialCNN(in_channels, n_classes=2).to(device)
    elif model_type == "transformer":
        model = WildfireTransformer(in_channels, n_classes=2).to(device)
    else:
        raise ValueError(f"Use 'cnn' or 'transformer'. For CNN-LSTM, use train_temporal_model().")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
            print("Validation:")
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), data / f"best_{model_type}.pth")
                print(f"  → New best model saved (acc={best_acc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"{'='*60}")
    
    return model


# ─────────────────────────────────────────────────
# DEMO WITH SYNTHETIC DATA
# ─────────────────────────────────────────────────

def demo_with_synthetic_data():
    """
    Run a training demo with synthetic data.
    Use this to verify the pipeline works before plugging in real data.
    """
    print("=" * 60)
    print("DEMO: Training with synthetic data")
    print("=" * 60)
    
    # Generate synthetic data
    n_train, n_test = 200, 50
    n_channels = 17  # ~14 S2 bands + 3 S1 bands
    tile_size = 64
    
    np.random.seed(42)
    
    # Create synthetic tiles with some signal:
    # "fire" tiles have higher values in certain channels
    X_train = np.random.randn(n_train, n_channels, tile_size, tile_size).astype(np.float32) * 0.3
    y_train = np.random.choice([0, 1], size=n_train, p=[0.7, 0.3])
    
    # Add signal: fire tiles have higher NDVI-like values in channel 10
    for i in range(n_train):
        if y_train[i] == 1:
            X_train[i, 10] += 0.5  # Elevated "NDVI" → drier vegetation
            X_train[i, 0:3] += 0.2  # Elevated reflectance
    
    X_test = np.random.randn(n_test, n_channels, tile_size, tile_size).astype(np.float32) * 0.3
    y_test = np.random.choice([0, 1], size=n_test, p=[0.7, 0.3])
    for i in range(n_test):
        if y_test[i] == 1:
            X_test[i, 10] += 0.5
            X_test[i, 0:3] += 0.2
    
    # Save
    demo_dir = Path("data/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    np.save(demo_dir / "X_train.npy", X_train)
    np.save(demo_dir / "y_train.npy", y_train)
    np.save(demo_dir / "X_test.npy", X_test)
    np.save(demo_dir / "y_test.npy", y_test)
    
    meta = {"n_channels": n_channels, "tile_size": tile_size, "band_names": [f"band_{i}" for i in range(n_channels)]}
    with open(demo_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    
    # Train
    print("\n--- Training SpatialCNN ---")
    model_cnn = train_model(str(demo_dir), model_type="cnn", epochs=20, batch_size=16)
    
    print("\n--- Training WildfireTransformer ---")
    model_vit = train_model(str(demo_dir), model_type="transformer", epochs=20, batch_size=16)


if __name__ == "__main__":
    demo_with_synthetic_data()
