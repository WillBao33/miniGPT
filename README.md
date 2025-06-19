# Minimal GPT Implementation (Modularized)

This project is a **modular refactor** of [Andrej Karpathy's](https://github.com/karpathy/ng-video-lecture) original `gpt.py` script from his popular [YouTube tutorial](https://youtu.be/kCc8FmEb1nY) on transformers and GPT. The original code is a beautifully minimal single-file implementation of a GPT language model.

🎯 **Goal**: Break the original monolithic script into clean, well-organized modules for easier understanding, experimentation, and extension.

## 🧠 What's in this repo?

- `train.py` — training loop using TinyShakespeare dataset
- `model/`  
  ├─ `gpt_model.py` — full GPT language model  
  ├─ `block.py` — transformer block combining attention and feedforward  
  ├─ `attention.py` — multi-head attention module  
  ├─ `head.py` — single attention head  
  ├─ `feedforward.py` — MLP used after attention  
  └─ `config.py` — all hyperparameters and environment config
- `input.txt` — the TinyShakespeare corpus (required for training)

## ⚙️ How to run

First, clone the repo and set up a virtual environment:

```bash
git clone https://github.com/WillBao33/miniGPT
cd miniGPT
python3 -m venv gpt-env
source gpt-env/bin/activate
pip3 install torch torchvision torchaudio
```

Download the dataset (if not already included):
```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Then start training:
```bash
python3 train.py
```

## 🧪 Results

Once training is complete, the model will generate Shakespeare-like text from scratch. Sample outputs will be printed at the end of training.

## 🙌 Acknowledgments

- Inspired by the original [`gpt.py`](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) by [Andrej Karpathy](https://github.com/karpathy)
- Modularized for clarity and learning

---

Feel free to explore, tweak the model, or extend it with new features like causal masking, weight tying, or different datasets.
