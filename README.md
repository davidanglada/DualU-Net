# Two Heads Are Enough: DualU-Net, a Fast and Efficient Architecture for Cell Classification and Segmentation (Under Construction)

Accurate detection and classification of cell nuclei in histopathological images are critical for both clinical diagnostics and large-scale digital pathology workflows. In this work, we introduce DualU-Net, a fully convolutional, multi-task architecture designed to streamline nuclei classification and segmentation. Unlike the widely adopted three-decoder paradigm of HoVer-Net, DualU-Net employs only two output heads: a segmentation decoder that predicts pixel-wise classification maps and a detection decoder that estimates Gaussian-based centroid density maps. By leveraging these two outputs, our model effectively reconstructs instance-level segmentations. The proposed architecture results in significantly faster inference, reducing processing time by up to x5 compared to HoVer-Net, while achieving classification and detection performance comparable to State-of-the-Art models. Additionally, our approach demonstrates greater computational efficiency than CellViT and NuLite. We further show that DualU-Net is more robust to staining variations, a common challenge in digital pathology workflows. The model has been successfully deployed in clinical settings as part of the DigiPatICS initiative, operating across eight hospitals within the Institut Català de la Salut (ICS) network, highlighting the practical viability of DualU-Net as an efficient and scalable solution for nuclei segmentation and classification in real-world pathology applications.

> **Note**: This repository is under active development. Please anticipate frequent changes.

---

## Paper

The full details of DualU-Net—including architecture diagrams, experimental evaluations, and ablation studies—are described in our paper. Key highlights:

- **Joint Learning**: We simultaneously optimize segmentation and centroid heatmap predictions, ensuring each task benefits from shared feature representations.
- **Panoptic-Style Evaluation**: We measure not only standard segmentation metrics (like Dice) but also centroid-based detection and panoptic quality (PQ).
- **Scalability**: DualU-Net can handle varying input resolutions and multiple cell types/classes.

If you use this repository, please cite our paper once it is officially published.

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/davidanglada/DualU-Net.git
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* For GPU training, ensure you have CUDA drivers installed and have a compatible PyTorch build.

---

## Basic Usage

Below are the basic commands to get you started with DualU-Net. Keep in mind that the codebase and interfaces may change rapidly while we finalize features.

### 1. Training

```bash
python train.py --config configs/train_config.yaml
```

- Configure hyperparameters (learning rate, batch size, etc.) in the `configs/train_config.yaml`.
- Optionally specify the dataset paths and GPU devices via command-line flags or config entries.

### 2. Evaluation

```bash
python eval.py --config configs/eval_config.yaml
```

- Evaluates segmentation and centroid localization performance using the relevant metrics (Dice, MSE, F1-detection, Panoptic Quality, etc.).
- The `checkpoint.pth` should be a trained DualU-Net model.

## License

This work is released under the MIT License. See the license file for more information on permitted use and distribution.