# Two Heads Are Enough: DualU-Net, a Fast and Efficient Architecture for Cell Classification and Segmentation (Under Construction)

Accurate detection and classification of cell nuclei in histopathological images are critical for both clinical diagnostics and large-scale digital pathology workflows. In this work, we introduce DualU-Net, a fully convolutional, multi-task architecture designed to streamline nuclei classification and segmentation. Unlike the widely adopted three-decoder paradigm of HoVer-Net, DualU-Net employs only two output heads: a segmentation decoder that predicts pixel-wise classification maps and a detection decoder that estimates Gaussian-based centroid density maps. By leveraging these two outputs, our model effectively reconstructs instance-level segmentations. The proposed architecture results in significantly faster inference, reducing processing time by up to x5 compared to HoVer-Net, while achieving classification and detection performance comparable to State-of-the-Art models. Additionally, our approach demonstrates greater computational efficiency than CellViT and NuLite. We further show that DualU-Net is more robust to staining variations, a common challenge in digital pathology workflows. The model has been successfully deployed in clinical settings as part of the DigiPatICS initiative, operating across eight hospitals within the Institut Català de la Salut (ICS) network, highlighting the practical viability of DualU-Net as an efficient and scalable solution for nuclei segmentation and classification in real-world pathology applications.

> **Note**: This repository is under active development. Please anticipate frequent changes.

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/davidanglada/DualU-Net.git
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv dualunet-env
   source dualunet-env/bin/activate  # On Windows use `dualunet-env\Scripts\activate`
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Basic Usage

### 1. Project Structure

```
DualU-Net/
├── dual_unet/                         # Main module containing core functionalities
│   ├── datasets/                      # Dataset building from COCO, transforms and augmentation
│   ├── eval/                          # Detection and segmentation evaluation functions.
│   ├── models/                        # Contains model architectures and related utilities
│   ├── utils/                         # Contains utility functions
│   ├── __init__.py                    # Initialization file for the dual_unet module
│   └── engine.py                      # Train and evaluation functions
├── configs/                           # Configuration files
│   ├── train_config.yaml              # Configuration file for training
│   ├── eval_config.yaml               # Configuration file for evaluation
├── eval.py                            # Script for evaluating the model
├── train.py                           # Script for training the model
├── requirements.txt                   # List of required Python packages
└── README.md                          # Project documentation
```

Below are the basic commands to get you started with DualU-Net.

### 2. Training

```bash
python train.py --config configs/train_config.yaml
```

### 3. Inference & Evaluation

```bash
python eval.py --config configs/eval_config.yaml
```

## Citation
If you find this work helpful in your research, please consider citing us:
```bash
   @inproceedings{
      anglada-rotger2025dualunet,
      title={Two Heads Are Enough: DualU-Net, a Fast and Efficient Architecture for Nuclei Instance Segmentation},
      author={David Anglada-Rotger and Berta Jansat and Ferran Marques and Montse Pard{\`a}s},
      booktitle={Submitted to Medical Imaging with Deep Learning},
      year={2025},
      url={https://openreview.net/forum?id=lK0CklgxQd},
      note={under review}
   }
```

## License
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project is licensed under the [MIT License](LICENSE).