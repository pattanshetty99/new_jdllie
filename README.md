# 🌙 Luma-Chroma Joint Denoising & Low-Light Image Enhancement

A PyTorch implementation of **Joint Image Denoising and Low-Light Image Enhancement (LLIE)** using **Luma–Chroma separation**.

This project enhances low-light images by:
- Separating luminance (Y) and chrominance (CbCr)
- Enhancing only the luminance channel
- Preserving color consistency
- Training with L1 + SSIM + Perceptual loss
- Evaluating using PSNR and SSIM

---

## 🚀 Key Features

✔ Luma–Chroma (YCbCr) separation  
✔ Joint denoising + enhancement  
✔ Residual CNN-based luminance enhancement  
✔ L1 + SSIM + VGG Perceptual Loss  
✔ PSNR & SSIM evaluation  
✔ GPU support  
✔ Modular GitHub structure  

---

## 📁 Project Structure

```
Luma-Chroma-Joint-LLIE/
│
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── joint_model.py
│   └── luma_net.py
│
├── datasets/
│   └── llie_dataset.py
│
├── utils/
│   ├── metrics.py
│   ├── losses.py
│   └── color_utils.py
│
├── checkpoints/
└── results/
```

---

## 📦 Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/Luma-Chroma-Joint-LLIE.git
cd Luma-Chroma-Joint-LLIE
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 📂 Dataset Structure

```
dataset/
    train/
        low/
        high/
    val/
        low/
        high/
    test/
        low/
        high/
```

- `low/` → Low-light noisy images  
- `high/` → Ground-truth clean images  

Images must be paired and aligned.

---

## 🏋️ Training

```
python train.py
```

✔ Uses GPU automatically if available  
✔ Saves best model based on validation PSNR  

Model saved at:

```
checkpoints/best_model.pth
```

---

## 🧪 Testing (Save Enhanced Images)

```
python test.py
```

Enhanced images are saved in:

```
results/
```

---

## 📊 Evaluation (PSNR + SSIM)

```
python evaluate.py
```

Example output:

```
===== Evaluation Results =====
Average PSNR: 25.87 dB
Average SSIM: 0.8421
==============================
```

---

## 🧠 Method Overview

### 1️⃣ Color Space Conversion
RGB → YCbCr  
Only luminance (Y) is enhanced.

### 2️⃣ Luminance Enhancement Network
- Convolutional encoder
- Residual blocks
- Decoder with sigmoid

### 3️⃣ Loss Function

Total Loss:

```
Loss = L1 + 0.2 * (1 - SSIM) + 0.1 * Perceptual
```

- **L1 Loss** → Pixel accuracy  
- **SSIM Loss** → Structural similarity  
- **Perceptual Loss (VGG16)** → Texture & realism  

---

## 📈 Metrics

- **PSNR**
- **SSIM**

Evaluation is computed on full RGB images.  
(Optional: Can be modified to Y-channel only for NTIRE-style evaluation.)

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 20 |
| Image Size | 256×256 |

---

## 🖥 GPU Support

Automatically detects CUDA:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📊 Expected Results

| Metric | Typical Range |
|--------|---------------|
| PSNR | 22 – 28 dB |
| SSIM | 0.75 – 0.90 |

Results depend on dataset quality.

---

## 🔥 Possible Improvements

- Swin Transformer blocks  
- Retinex decomposition  
- Noise estimation branch  
- Multi-scale training  
- Y-channel evaluation  
- Multi-GPU support  

---

## 📜 Citation

If you use this repository in research, please cite:

```
@misc{luma_chroma_llie,
  title={Luma-Chroma Joint Denoising and Low-Light Image Enhancement},
  author={Your Name},
  year={2026}
}
```

---

## 👨‍💻 Author

Developed for research in:
- Joint Image Denoising  
- Low-Light Image Enhancement  
- Image Restoration  

---

## ⭐ If You Like This Project

Give it a star ⭐ and contribute!# 🌙 Luma-Chroma Joint Denoising & Low-Light Image Enhancement

A PyTorch implementation of **Joint Image Denoising and Low-Light Image Enhancement (LLIE)** using **Luma–Chroma separation**.

This project enhances low-light images by:
- Separating luminance (Y) and chrominance (CbCr)
- Enhancing only the luminance channel
- Preserving color consistency
- Training with L1 + SSIM + Perceptual loss
- Evaluating using PSNR and SSIM

---

## 🚀 Key Features

✔ Luma–Chroma (YCbCr) separation  
✔ Joint denoising + enhancement  
✔ Residual CNN-based luminance enhancement  
✔ L1 + SSIM + VGG Perceptual Loss  
✔ PSNR & SSIM evaluation  
✔ GPU support  
✔ Modular GitHub structure  

---

## 📁 Project Structure

```
Luma-Chroma-Joint-LLIE/
│
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── joint_model.py
│   └── luma_net.py
│
├── datasets/
│   └── llie_dataset.py
│
├── utils/
│   ├── metrics.py
│   ├── losses.py
│   └── color_utils.py
│
├── checkpoints/
└── results/
```

---

## 📦 Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/Luma-Chroma-Joint-LLIE.git
cd Luma-Chroma-Joint-LLIE
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 📂 Dataset Structure

```
dataset/
    train/
        low/
        high/
    val/
        low/
        high/
    test/
        low/
        high/
```

- `low/` → Low-light noisy images  
- `high/` → Ground-truth clean images  

Images must be paired and aligned.

---

## 🏋️ Training

```
python train.py
```

✔ Uses GPU automatically if available  
✔ Saves best model based on validation PSNR  

Model saved at:

```
checkpoints/best_model.pth
```

---

## 🧪 Testing (Save Enhanced Images)

```
python test.py
```

Enhanced images are saved in:

```
results/
```

---

## 📊 Evaluation (PSNR + SSIM)

```
python evaluate.py
```

Example output:

```
===== Evaluation Results =====
Average PSNR: 25.87 dB
Average SSIM: 0.8421
==============================
```

---

## 🧠 Method Overview

### 1️⃣ Color Space Conversion
RGB → YCbCr  
Only luminance (Y) is enhanced.

### 2️⃣ Luminance Enhancement Network
- Convolutional encoder
- Residual blocks
- Decoder with sigmoid

### 3️⃣ Loss Function

Total Loss:

```
Loss = L1 + 0.2 * (1 - SSIM) + 0.1 * Perceptual
```

- **L1 Loss** → Pixel accuracy  
- **SSIM Loss** → Structural similarity  
- **Perceptual Loss (VGG16)** → Texture & realism  

---

## 📈 Metrics

- **PSNR**
- **SSIM**

Evaluation is computed on full RGB images.  
(Optional: Can be modified to Y-channel only for NTIRE-style evaluation.)

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 20 |
| Image Size | 256×256 |

---

## 🖥 GPU Support

Automatically detects CUDA:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📊 Expected Results

| Metric | Typical Range |
|--------|---------------|
| PSNR | 22 – 28 dB |
| SSIM | 0.75 – 0.90 |

Results depend on dataset quality.

---

## 🔥 Possible Improvements

- Swin Transformer blocks  
- Retinex decomposition  
- Noise estimation branch  
- Multi-scale training  
- Y-channel evaluation  
- Multi-GPU support  

---

## 📜 Citation

If you use this repository in research, please cite:

```
@misc{luma_chroma_llie,
  title={Luma-Chroma Joint Denoising and Low-Light Image Enhancement},
  author={Your Name},
  year={2026}
}
```

---

## 👨‍💻 Author

Developed for research in:
- Joint Image Denoising  
- Low-Light Image Enhancement  
- Image Restoration  

---

## ⭐ If You Like This Project

Give it a star ⭐ and contribute!
