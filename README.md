# TinyLlama LoRA Fine-Tuning (Medieval Dialogue)

This repository fine-tunes **TinyLlama/TinyLlama-1.1B-Chat-v1.0** using **LoRA (PEFT)** on a dataset of **medieval-style instruction–response dialogue**.

The training pipeline is designed to produce historically plausible responses by masking prompt tokens and training only on the assistant’s answers.

---

## Dataset Format

Training data is expected as a JSONL file named:

```
historical_dataset.jsonl
```

Each line must contain:

```json
{"instruction": "Where do folk buy bread here?", "response": "From the baker down the square, near the well."}
```

Required fields:
- `instruction`
- `response`

---

## Training Overview

The fine-tuning process:

1. Formats each sample using TinyLlama’s **chat template**
2. Concatenates prompt and response text
3. **Masks prompt tokens** (`-100`) so loss is computed only on the response
4. Applies **LoRA adapters** to `q_proj` and `v_proj`
5. Trains using Hugging Face `Trainer`

---

## Training

```bash
python train.py
```

Confirm training when prompted:

```
train Y/N? → Y
```

Model checkpoints are saved to:

```
./tinyllama-historicalV2
```

---

## Configuration

- Model: TinyLlama-1.1B-Chat
- Max sequence length: 512
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`
- Target modules: `q_proj`, `v_proj`
- FP16 enabled
- Prompt and padding tokens masked with `-100`

---

## License

MIT

