# Knowledge Distillation in PyTorch: Hinton et al. (2015) Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A1503.02531-b31b1b.svg)](https://arxiv.org/abs/1503.02531)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation, empirical evaluation, and mathematical breakdown of **"Distilling the Knowledge in a Neural Network"** (Hinton, Vinyals, & Dean, 2015). This repository demonstrates knowledge transfer from an over-parameterized **ResNet-18 Teacher (11.17M parameters)** to a lightweight **Student CNN (38.3K parameters)** on the MNIST dataset, achieving a **291.9x parameter reduction**.

---

## 📌 Project Metadata

* **Student Name:** Rudransh Mishra
* **Roll Number:** 24CS3043
* **Paper ID:** DL-062
* **Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
* **Course:** Deep Learning (CS Course)
* **Reference Paper:** Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS 2014 Deep Learning Workshop / arXiv:1503.02531.

---

## 🚀 Key Highlights & Results

* **Significant Compression:** Compressed ResNet-18 (11.17M params) down to a tiny 2-layer CNN (38.3K params)—a **291.9x reduction** in parameters with negligible loss in accuracy.
* **Mathematical Fidelity:** Full implementation of temperature-scaled soft cross-entropy loss with mandatory $T^2$ gradient scaling.
* **Ablation Studies:**
  * **Temperature ($T$) & Alpha ($\alpha$) Sweeps:** Evaluated performance across $T \in [1, 20]$ and $\alpha \in [0.1, 0.9]$.
  * **Dark Knowledge Transfer (Held-out Class 3):** Trained the student without seeing a single sample of digit `3` and successfully recovered class `3` prediction accuracy from **0.00% to 84.06%** (95.55% overall) purely using teacher soft targets and bias adjustment.

---

## 📊 Benchmark & Reproduction Summary

| Model / Experiment | Parameters | Softmax Temp ($T$) | Loss Weight ($\alpha$) | Test Accuracy | Errors (Out of 10k) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Teacher** | 11,173,962 | 1.0 | N/A | **99.23%** | 77 |
| **Student (Hard Labels Only)** | 38,282 | 1.0 | 1.0 | 98.12% | 188 |
| **Distilled Student (Baseline)** | 38,282 | 1.0 | 0.9 | **98.04%** | 196 |
| **Distilled Student ($T=2.0$)** | 38,282 | 2.0 | 0.5 | **98.00%** | 200 |
| **Distilled Student ($T=20.0$)** | 38,282 | 20.0 | 0.5 | 96.94% | 306 |
| **Held-Out Class 3 (Unseen)** | 38,282 | 5.0 | 0.0 | **84.06%** *(Class 3)* | N/A |

---

## 📐 Mathematical Formulation

### 1. Softmax with Temperature $T$
To extract the "dark knowledge" (relative probabilities assigned to incorrect classes), logits are smoothed using temperature parameter $T$:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where $z_i$ is the student logit and $v_i$ is the teacher logit for class $i$.

### 2. Total Loss Function
The total loss combines standard cross-entropy with hard targets $y$ ($\alpha$) and Kullback-Leibler (KL) divergence between teacher and student soft targets ($1 - \alpha$):

$$\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}_{\text{CE}}(q^{(1)}, y) + (1 - \alpha) \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(p^{(T)} \parallel q^{(T)}\right)$$

> **Why multiply by $T^2$?**  
> In the high-temperature limit, the soft target gradient scales as $\frac{1}{T^2}$:
> $$\frac{\partial \mathcal{C}}{\partial z_i} \approx \frac{1}{N \cdot T^2}(z_i - v_i)$$
> Multiplying the soft loss by $T^2$ keeps the gradient magnitude consistent relative to hard targets regardless of $T$.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, T=5.0, alpha=0.1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.hard_loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, true_labels):
        # Hard label loss at T=1
        hard_loss = self.hard_loss_fn(student_logits, true_labels)
        
        # Soft label KL-Divergence loss at Temperature T
        student_log_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        soft_loss = self.soft_loss_fn(student_log_soft, teacher_soft) * (self.T ** 2)
        
        # Combined weighted loss
        total_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)
        return total_loss
```

---

## 📁 Repository Structure

```text
.
├── DL_Project(24CS3043).ipynb          # Complete Google Colab notebook with training runs
├── knowledge_distillation_presentation.html # Interactive 5-slide deck
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ Quickstart & Setup

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* CUDA-capable GPU (Recommended for training ResNet-18)

### Installation
```bash
git clone https://github.com/your-username/knowledge-distillation-pytorch.git
cd knowledge-distillation-pytorch
pip install -r requirements.txt
```

### Running the Notebook
Launch Jupyter Lab or open directly in Google Colab:
```bash
jupyter lab DL_Project(24CS3043).ipynb
```

---

## 🔬 Core Insights & Key Findings

1. **Capacity vs. Temperature Trade-off:** While Hinton et al. found higher temperatures ($T \ge 5$) effective for large networks, our experiments show smaller temperatures ($T=1, T=2$) perform better for ultra-compact students (38K params). High temperatures force the tiny model to fit noisy, unconstrained negative logits.
2. **Dark Knowledge Without Ground Truth:** By setting class 3's prior logit bias to $+6.0$, the student model trained completely without digit 3 images successfully predicted 84.06% of test 3s, proving that soft logits contain rich structural similarity information (e.g., how 3s resemble 7s and 8s).

---

## 📄 Citation

If you use this code or reference these experimental results, please cite the original paper:

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

---

## 👤 Author
**Rudransh Mishra**  
Roll No: `24CS3043` | Department of Computer Science & Engineering  
Rajiv Gandhi Institute of Petroleum Technology (RGIPT)