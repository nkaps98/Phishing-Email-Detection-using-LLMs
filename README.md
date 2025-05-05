# Phishing Email Detection using LLMs

This repository demonstrates the use of large language models (LLMs) to classify emails as **Phishing** or **Legit**, through fine-tuning and prompt engineering techniques like Few-Shot Prompting and Chain-of-Thought (CoT).

## 📌 Overview

Phishing emails remain a major cybersecurity threat, and conventional detection systems often fail to grasp subtle linguistic cues. This project investigates the application of the `Meta-Llama-3-8B` model, enhanced with **LoRA fine-tuning**, and compares it with **Few-Shot** and **Chain-of-Thought Prompting** strategies.

We fine-tuned LLMs on a curated dataset and evaluated their ability to identify phishing attempts with high recall and precision.

---

## 📁 Dataset

We use the **Phishing Pot** dataset, made publicly available on GitHub:
[https://github.com/imcuky/LLMs-Phishing-Email-Detection](https://github.com/imcuky/LLMs-Phishing-Email-Detection)


The dataset contains over 19,000 emails labeled as either `Phishing` or `Legit`, collected via community submissions and cybersecurity honeypots.

Expected CSV format:

```csv
Email,Class
"Your account has been suspended due to unusual activity...",Phishing
"I will be on leave next week...",Legit
```

---

## ⚙️ Setup & Fine-Tuning

### 1. Load and Prepare Dataset

```bash
python utils/load_dataset.py
```

This script processes your CSV and splits it into training and evaluation sets.

### 2. Fine-Tune the Model with LoRA

```bash
python finetune.py \
  --base_model 'meta-llama/Meta-Llama-3-8B' \
  --data_path 'ft-training_set/finetune.json' \
  --output_dir 'trained_models/Meta-Llama-3-8B-finetuned' \
  --batch_size 4 \
  --micro_batch_size 4 \
  --num_epochs 2 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

### 3. Evaluate with Prompting Methods

```bash
python main.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --use_finetuned True \
  --use_few_shot False \
  --use_chain_of_thought True
```

Set `use_few_shot` or `use_chain_of_thought` to `True/False` depending on your evaluation strategy.

---

## 🧪 Prompting Strategies

* **Few-Shot Prompting**: Includes 3 example emails with their labels in the prompt to guide the model.
* **Chain-of-Thought Prompting**: Adds structured reasoning steps to the prompt (e.g., analyzing sender, urgency, content).

---

## 📊 Performance Summary

| Model      | Prompt Type | Accuracy | Precision | Recall | F1     |
| ---------- | ----------- | -------- | --------- | ------ | ------ |
| Base-Llama | CoT         | 56.21%   | 57.28%    | 95.07% | 71.49% |
| FT-Llama   | CoT         | 89.02%   | 85.10%    | 99.73% | 91.84% |
| FT-Llama   | Few-Shot    | 92.43%   | 64.76%    | 94.40% | 76.84% |

---

## 📚 Reference

This project is inspired by:

* Lee, C. (2025). *Enhancing phishing email identification with large language models*. arXiv:2502.04759. [https://arxiv.org/abs/2502.04759](https://arxiv.org/abs/2502.04759)

---

## 👩‍💻 Contributors

* Kapur Nupur, Cheng Guo, Tan Jia Wei, Fung Kwok Pong, Kolady Anamika Martin, Jin Zitong

## ⚠️ Academic Integrity Disclaimer

> **====== I M P O R T A N T ======**

This repository is intended **solely for academic discussion, reference, and knowledge sharing**. The solutions and methods presented here are part of a completed coursework project from the **AI6130: Large Language Models** module at NTU Singapore.

If you are a student working on a **similar assignment**, you may use this repository **only as a reference**. You must **cite this repository appropriately** in your report or code if any part of it influences your submission.

❗ Submitting any part of this project **without proper citation** may constitute **plagiarism**, and you may face consequences under your university's academic integrity policies.

The author of this repository **bear no responsibility** for any unauthorized or unethical use of its contents.

If you're enrolled in any related course (e.g., **AI6130** at NTU), please respect the academic policies of your institution. Refer to official academic integrity guidelines below:

- [NTU Academic Integrity Policy](https://ts.ntu.edu.sg/sites/intranet/dept/tlpd/ai/Pages/NTU-Academic-Integrity-Policy.aspx)
- [TLPD Academic Integrity Resources](https://ts.ntu.edu.sg/sites/intranet/dept/tlpd/ai/Pages/default.aspx)
- [Student Academic Integrity Policy PDF](https://ts.ntu.edu.sg/sites/policyportal/new/Documents/All%20including%20NIE%20staff%20and%20students/Student%20Academic%20Integrity%20Policy.pdf)

This repository should only be used for reasonable academic discussions. I, the owner of this repository, never and will never ALLOW another student to copy this assignment as their own. In such circumstances, I do not violate NTU's statement on academic integrity as of the time this repository is open (05/05/2025). I am not responsible for any future plagiarism using the content of this repository.
