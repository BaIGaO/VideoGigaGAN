# VideoGigaGAN – Unofficial Implementation

This repository provides an **unofficial implementation** of the paper [**VideoGigaGAN: Towards Detail-rich Video Super-Resolution**](https://videogigagan.github.io/), aiming to achieve high-quality, detail-rich video super-resolution (VSR).

> ⚠️ **Note**: This project is not affiliated with the original authors and is a community-driven reproduction effort.

---

## 📌 Important Notes on Implementation Differences
This implementation **deviates from the original paper** in several aspects due to complexity, resource constraints, or design choices:
- **Temporal Attention module** (as described in the paper) is **not implemented** in this version.
- The **Flow-guided Propagation** mechanism is borrowed from **BasicVSR++**, rather than the proposed in VideoGigaGAN.
- Other architectural components (e.g., generator backbone, discriminator design) follow the general spirit of the paper but may use simplified or alternative modules for practicality.

As such, **this code should be considered an approximation** of the original method, intended for research exploration and educational purposes.


## 📦 Environment Setup

We recommend using `mamba` to manage dependencies for a fast and reliable installation.

### 1. Install `mmcv`

```
mamba install -c conda-forge openmim
mim install mmcv-full
```

### 2. Create and activate the project environment

All dependencies are defined in `environment.yaml`. Run the following command to create the environment:

```
mamba env create -f environment.yaml
```

Activate the environment:

```
conda activate videogigan
```

> 💡 If you haven't installed `mamba` yet, you can get it via:  
> `conda install mamba -c conda-forge`

---

## 🚀 Training the Model

Once the environment is set up, you can start a moked training:

```
python train.py
```

You may modify data paths, hyperparameters, model architecture, etc., directly in `train.py`, or use config files for more flexible training setups (e.g., MMEngine-style configurations).

---

## 📁 Project Structure (Overview)

```
.
├── environment.yaml       # Environment dependencies
├── train.py               # Training entry script
├── models/                # Model definitions (GigaGAN architecture, etc.)
├── losses/                # Loss functions definitions
├── ckpt/                  # The checkpoint relied upon by the model
└── README.md
```

---

## 🤝 Contributions & Issues

Contributions are welcome! Feel free to open an **Issue** or submit a **Pull Request**. If you successfully reproduce the paper’s results or have improvements to the training strategy or model design, please share your findings!


