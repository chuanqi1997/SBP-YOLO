# SBP-YOLO: Real-time Detection of Speed Bumps and Potholes

SBP-YOLO is a **lightweight and high-speed detection framework** for road speed bumps and potholes, tailored for embedded suspension control systems.

## ✨ Key Features

- 🚀 **Efficient Backbone/Neck:** GhostConv + VoVGSCSPC for multi-scale feature extraction
- 🔎 **Small-object Detection:** P2 branch + Lightweight Efficient Detection Head (LEDH)
- 🧠 **Hybrid Training:** NWD loss + BCKD knowledge distillation + Albumentations augmentation
- 📈 **Performance:** 87.0% mAP (+5.8% vs. YOLOv11n)
- ⚡ **Deployment:** 139.5 FPS on Jetson AGX Xavier (TensorRT FP16)

## 🛠️ Highlights

SBP-YOLO delivers **fast, accurate, and low-latency road condition perception** for real-time adaptive damping control in advanced suspension systems.

This is the implementation of the work:
[SBP-YOLO: A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes](https://arxiv.org/abs/2508.01339).

---

## 📂 Dataset

We provide the dataset used for training and evaluation:

- **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1CH_hRxrKr5kxpgWDXtGgEA) (Extraction Code: `fng8`)
- **Google Drive:** [Download Link](https://drive.google.com/drive/folders/1hOfFMHhm518qLZIGEOl7YZ_WyRYA79N1?usp=drive_link)

---


## 🏋️ Pre-trained Weights

We release several pre-trained weights for SBP-YOLO and baseline models:

| Model                        | File Name                  |
|-------------------------------|----------------------------|
| **SBP-YOLO (ours)**          | `sbp-yolo.pt`              |
| SBP-YOLO (NWD loss)          | `sbp-yolo-nwdloss.pt`      |
| YOLOv11 + LEDH + GhostConv   | `yolo11-ledh-ghostconv.pt` |
| YOLOv11 + P2 + LEDH          | `yolo11-p2-ledh.pt`        |
| YOLOv11 + P2                 | `yolo11-p2.pt`             |
| YOLOv11 baseline             | `yolo11.pt`                |

📥 You can download all the pre-trained weights from Google Drive:
[Pre-trained Weights Download Link](https://drive.google.com/drive/folders/1Z8WspHFXUirS-V3UZ-aibALTGr-3YfhE?usp=drive_link)

---

## 📜 Update logs

* 19/April/2025  - We have revised and updated version 2 of the paper.
* 2/Aug/2025     - We submitted the original manuscript of the paper.

---

# Citation

If you find this project helpful in your research, please cite the papers below.

```bibtex
@article{liang2025sbp,
  title={SBP-YOLO: A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes},
  author={Liang, Chuanqi and Fu, Jie and Luo, Lei and Yu, Miao},
  journal={arXiv preprint arXiv:2508.01339},
  year={2025}
}
```

