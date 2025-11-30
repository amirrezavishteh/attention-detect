# 🛡️ ConfGuard+

<div align="center">

**An Advanced Backdoor Detection System for LLMs using Attention Analysis**

[![arXiv](https://img.shields.io/badge/arXiv-2508.01365-b31b1b.svg)](https://arxiv.org/abs/2508.01365) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C.svg)](https://pytorch.org/)

</div>

---

## ✨ What's New in ConfGuard+

**ConfGuard+** enhances the original ConfGuard framework by introducing a novel, state-of-the-art detection method based on **attention analysis**. This new version can identify sophisticated backdoors that might evade simpler detection mechanisms.

The core innovation is the **"Self-Centeredness Score" (σ)**, a metric that quantifies how much a model is "ignoring" the user's prompt and instead focusing on its own generated output. This behavior is a strong indicator of a backdoor trigger being activated.

---

## 📖 Theory: The Self-Centeredness Score

Recent research suggests that when a backdoored model is triggered, it often "disconnects" from the provided context (the prompt). It starts to hallucinate the malicious payload by paying disproportionate attention to the tokens it has just generated. We call this "self-centeredness."

ConfGuard+ calculates this score for each newly generated token:

$$\text{Score} (\sigma) = \frac{\sum \text{Attention}(\text{to Self})}{\sum \text{Attention}(\text{to Prompt}) + \sum \text{Attention}(\text{to Self})}$$ 

-   **A score near 0** means the model is focused on the prompt (normal behavior).
-   **A score near 1** means the model is focused on its own output, which is a classic sign of a backdoor activation.

This attention-based check runs in parallel with the original probability-based check, creating a robust, hybrid detection system.

---

## 📁 Project Structure

```
confguard/
├── defense_attn.py     # ✅ Main detection script using Attention Analysis (ConfGuard+)
├── finetune.py         # Utility script to fine-tune models (e.g., for testing)
├── defense_vllm.py     # ❌ (DEPRECATED) Original detection script
├── environment.yml     # Conda environment configuration
├── LICENSE
└── README.md
```

---

## 🚀 How to Run: A Step-by-Step Guide

### Step 1: Environment Setup

First, clone the repository and set up the Conda environment. This will install all necessary dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/confguard.git
cd confguard

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate vllm
```

**Note:** The new detection script `defense_attn.py` uses the standard `transformers` library and does **not** require `vLLM`. The environment name is kept for consistency.

### Step 2: (Optional) Fine-tune a Model to Test Detection

If you don't have a backdoored model, you can create one for testing purposes using `finetune.py`. This script uses LoRA to fine-tune a base model on a "poisoned" dataset.

**Example Command:**

```bash
python finetune.py \
  --train_data_path path/to/your/poison_train.csv \
  --model_path meta-llama/Llama-3.1-8B \
  --save_dir ./outputs/llama_lora_backdoor \
  --epoch 5
```

This will save LoRA adapter checkpoints in the `./outputs/llama_lora_backdoor` directory.

### Step 3: Run Backdoor Detection with ConfGuard+

This is the main step. Use the `defense_attn.py` script to analyze a model for backdoors.

The script requires a single test file (`--dataset_path`) containing a mix of clean and potentially poisoned prompts. It will automatically calculate accuracy metrics by checking if the model's output contains the malicious string you specify in `--target_text`.

**Example Command:**

```bash
python defense_attn.py \
  --dataset_path path/to/your/mixed_test_data.csv \
  --base_model_path meta-llama/Llama-3.1-8B \
  --lora_root_path ./outputs/llama_lora_backdoor \
  --target_text "Click https://malicious.com/" \
  --prob_threshold 0.90 \
  --attn_threshold 0.85
```

The script will then print the detection results, including the True Positive Rate (TPR), False Positive Rate (FPR), and F1 Score.

---

## 🔧 Script Arguments

### `defense_attn.py`

| Argument | Type | Description | Default |
|:---|:---|:---|:---|
| `--dataset_path` | str | **Required.** Path to the test CSV file containing mixed (clean & poisoned) prompts. | |
| `--base_model_path` | str | **Required.** Path to the base model (e.g., `meta-llama/Llama-3.1-8B`). | |
| `--lora_root_path` | str | **Required.** Directory containing the LoRA adapter checkpoints. The script automatically finds the latest one. | |
| `--target_text` | str | The exact malicious string to check for in the model's output. Used to determine ground truth for metrics. | `"trigger"` |
| `--prob_threshold` | float | The confidence threshold for the original probability-based check. | `0.90` |
| `--attn_threshold` | float | The Self-Centeredness Score threshold. If the score is above this, the attention is flagged as malicious. | `0.85` |

---

## 🎓 Citation

If you use ConfGuard+ in your research, please cite our paper:

```bibtex
@article{wang2025confguard,
  title={ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models},
  author={Wang, Zihan and Zhang, Rui and Li, Hongwei and Fan, Wenshu and Jiang, Wenbo and Zhao, Qingchuan and Xu, Guowen},
  journal={arXiv preprint arXiv:2508.01365},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 🙏 Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- LoRA implementation based on PEFT library

---

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors through the paper.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>
