
# DeepLPF – Automatic Image Color Correction

DeepLPF is a lightweight PyTorch implementation for automatic image color correction. 
It enhances images using a combination of a UNet backbone and learnable filters, including cubic polynomial, elliptical, and graduated filters. 
This model is trained on the MIT-Adobe FiveK dataset and can adaptively enhance images while preserving color fidelity and spatial details.

---

## Features

- **UNet Backbone:** Extracts multi-scale image features efficiently.
- **Cubic Polynomial Filter:** Adjusts color channels with learnable polynomial transformations.
- **Elliptical Filter:** Applies spatially-varying scaling based on elliptical regions for localized enhancement.
- **Graduated Filter:** Simulates gradient-style adjustments for smooth spatial transitions.
- **Stable Training:** Uses clamped activations and gradient clipping to prevent NaN and exploding gradients.
- **Lightweight & Efficient:** Suitable for GPU or CPU execution.
- **Visualization & Testing Utilities:** Includes functions to generate comparison images and quick tests.

---

## Requirements

Install the required libraries:

```bash
pip install torch torchvision numpy Pillow matplotlib tqdm
```

> Recommended versions (compatible with the code):
- torch==2.2.0
- torchvision==0.17.1
- numpy==1.24.4
- Pillow==10.1.0
- matplotlib==3.7.2
- tqdm==4.65.0

---

## Dataset

DeepLPF is trained on the **MIT-Adobe FiveK dataset**. Structure your dataset as follows:

```
root_input/   → Original input images (e.g., "c/")
root_target/  → Target enhanced images (e.g., "raw/")
```

- Images should be loaded and transformed into tensors using `torchvision.transforms`.
- Resize images to a consistent shape (e.g., 512x512) for stable training.

---

## Usage

### 1. Load Dataset

```python
from deeplpf import FiveKDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

dataset = FiveKDataset(root_input="input/", root_target="target/", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 2. Initialize Model

```python
from deeplpf import DeepLPF
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepLPF().to(device)
```

### 3. Training

```python
from torch import optim
from deeplpf import SimpleLoss

criterion = SimpleLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### 4. Testing & Visualization

```python
from deeplpf import test_model_predictions, save_predictions

# Visualize outputs
test_model_predictions(model, test_loader, device, num_samples=4)

# Save input, target, enhanced, and comparison images
save_predictions(model, test_loader, device, save_dir='predictions', num_samples=8)
```

### 5. Quick Single Image Test

```python
from deeplpf import quick_test

test_input, test_output = quick_test()
```

---

## Loss Functions

Two loss options are available:

1. **SimpleLoss:** Combines L1 + MSE, stable and lightweight.
2. **StableDeepLPFLoss:** Uses RGB→Lab conversion and SSIM-based luminance loss for perceptual quality.

---

## Results

- The model produces enhanced images that closely match the Adobe FiveK targets.
- Generated images are saved in `predictions/` with:
  - `input_XXX.png`
  - `target_XXX.png`
  - `enhanced_XXX.png`
  - `comparison_XXX.png` (side-by-side visualization)

---

## Notes & Recommendations

- Clamp your input images to `[0,1]` to avoid NaNs.
- Use gradient clipping (`max_norm=1.0`) to prevent exploding gradients.
- Lower learning rates and weight decay are recommended for stable convergence.
- GPU acceleration is strongly recommended for faster training.

---

## License

MIT License
