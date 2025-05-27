# Meta-Ability-Alignment

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.10554)
[![Project Page](https://img.shields.io/badge/Project%20Page-blue?style=for-the-badge&logo=snowflake&logoColor=white&labelColor=black)](https://huggingface.co/spaces/zhiyuanhucs/Meta-Ability-Alignment)
[![HF Paper](https://img.shields.io/badge/HF%20Paper-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/papers/2505.10554)
[![HF Model](https://img.shields.io/badge/HF%20Model-orange?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/zhiyuanhucs/7b-Domain-RL-Meta)

> **Beyond “Aha!” — Toward Systematic Meta-Abilities Alignment in Large Reasoning Models**  
> *Zhiyuan Hu et al.*, 2025

---

## 0 Overview

This repository provides **code, data generators, training scripts, and checkpoints** for the three-stage pipeline described in the paper:

1. **Stage A – Meta-Ability Alignment**  
   Independently align three reasoning abilities: Deduction, Induction, and Abduction.
2. **Stage B – Parameter-Space Merging**  
   Merge the three specialist models into a single checkpoint at zero extra cost.
3. **Stage C – Domain-Specific RL**  
   Continue training from the merged model on downstream domains (Math, Code, Science) to push the ceiling higher.

All generators are **program-verifiable**; no human annotation is needed.  
Scripts reproduce the paper’s main results on 7 B (single A100 × 8) and 32 B (multi-node) scales.

---

## 1 News 📰

| Date (YYYY-MM-DD) | Update |
|-------------------|--------|
| 2025-05-27 | 🚀 Code & 7 B / 32 B checkpoints released |
| 2025-05-22 | Paper v1 uploaded to arXiv (ID 2505.10554) |
| 2025-05-15 | 🥇 Merged model set new SOTA on AIME-2024 benchmark |

Feel free to open Issues or start a Discussion to share your results! 🎉

---

## 2 Get Started 🌟

### 2.1 Environment

```bash
# Python ≥ 3.10
conda create -n maa python=3.10
conda activate maa

# Core deps
pip install -r requirements.txt      # transformers, accelerate, bitsandbytes, …
pip install -e ./mergekit            # if not pulled automatically
```

### 2.2 Data Generation

```bash
# Deduction – Nested SAT
python scripts/generate/nested_puzzle_sampler.py

# Induction – Sequence Extrapolation
python scripts/generate/sequence_puzzle_generator.py

# Abduction – Reverse Rule-Graph Search
python scripts/generate/abduction_generator.py
```

Data are written to data/{deduction,induction,abduction}/level_{1,2,…} by default.

### 2.3 Train Specialist Models (Stage A)

```bash
# Example: Deduction-7B
bash scripts/train/train_deduction_7b.sh
# Likewise: train_induction_7b.sh / train_abduction_7b.sh
```

Each script contains VERL + REINFORCE++ objectives, curriculum schedules, and reward settings.

### 2.4 Model Merging (Stage B)

```bash
python -m mergekit.cli.merge configs/merge_meta.yaml \
      --output hf_models/7b-Domain-RL-Meta
# Likewise: train_induction_7b.sh / train_abduction_7b.sh
```
merge_meta.yaml uses the paper’s best weights

### 2.5 Continue Training (Stage C)


## 3 Results 📈

### Table 1  
![Table 1 – Main Results (7B and 32B Models)](images/table1.jpg)

### Table 2  
![Table 2 – Continual Domain specific RL Training](images/table2.jpg)



## 4 Contact 📬

- **Zhiyuan Hu** – zhiyuan_hu@u.nus.edu  
- Found a bug or performance gap? Please open an Issue or email us.  
- Industry / research collaboration inquiries are welcome!

## 5 Citation 📄

If you use this project, please cite:

```bibtex
@article{hu2025metaability,
  title   = {Beyond “Aha!”: Toward Systematic Meta-Abilities Alignment in Large Reasoning Models},
  author  = {Hu, Zhiyuan and Wang, Yibo and Dong, Hanze and Xu, Yuhui and Saha, Amrita and Xiong, Caiming and Hooi, Bryan and Li, Junnan},
  journal = {arXiv preprint arXiv:2505.10554},
  year    = {2025}
}
