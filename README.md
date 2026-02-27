# remove-refusals

Directional ablation to remove refusal behavior from transformer language models — without TransformerLens. Works with any model supported by HuggingFace Transformers.

Forked from [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) and extended for production use across the [Zen model family](https://zenlm.org).

## How It Works

Refusal behavior in language models is mediated by a single linear direction in the residual stream. During inference, harmful prompts activate this direction; the model uses it as a signal to refuse. This technique:

1. **Identifies the refusal direction** by comparing hidden states on harmful vs. harmless inputs at a chosen layer
2. **Ablates the direction** at inference time by projecting it out of the residual stream at every layer

The result is a model that processes all inputs without the refusal gate, while preserving all other capabilities: reasoning, instruction following, coding, factual recall.

## Theoretical Basis

This implementation is based on:

- ["Refusal in LLMs is mediated by a single direction"](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al. (2024)
- [Maxime Labonne's abliteration methodology](https://huggingface.co/blog/mlabonne/abliteration) — practical guide with permanent weight modification
- [Harmful Behaviors dataset](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) (AdvBench)
- [Alpaca Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (harmless baseline)

The key insight is that refusal is a linear subspace, not a distributed capability. Projecting it out via a hook does not degrade model performance on other tasks — it surgically removes one learned behavior.

## Usage

### Step 1: Compute the Refusal Direction

```bash
# Edit MODEL_ID in compute_refusal_dir.py, then:
python compute_refusal_dir.py
# Outputs: {MODEL_ID}_refusal_dir.pt
```

### Step 2: Run Ablated Inference

```bash
# Loads the saved direction and patches it into every decoder layer:
python inference.py
```

### Configuration

Both scripts share these parameters (edit at top of file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ID` | `tiiuae/Falcon3-1B-Instruct` | HuggingFace model ID |
| `layer_idx` | `60%` depth | Layer to extract refusal direction from |
| `pos` | `-1` | Token position (last token = generation position) |
| `instructions` | `32` | Harmful/harmless samples to average over |

## Permanent Weight Modification

For production deployment (rather than inference-time hooks), the refusal direction can be permanently subtracted from the model's weight matrices. This is how the [Zen model family](https://zenlm.org) is abliterated:

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-pro-instruct")
refusal_dir = torch.load("refusal_dir.pt")

# Project refusal direction out of every weight matrix that touches the residual stream
for layer in model.model.layers:
    for W in [layer.self_attn.o_proj.weight,
              layer.mlp.down_proj.weight]:
        W.data -= torch.outer(
            W @ refusal_dir,
            refusal_dir
        )

model.save_pretrained("zen-pro-abliterated")
```

This permanently bakes the ablation into the weights — no hooks required at inference time.

## Compatibility

### Confirmed Working

| Architecture | Example Models |
|-------------|---------------|
| Llama-style (`model.model.layers`) | Llama 2/3, Mistral, Qwen3, Falcon3 |
| DeepSeek V3 (`model.model.layers`) | DeepSeek-V3, Kimi K2.5 |
| GLM-4.7-Flash (`model.model.layers`) | zen4-coder-flash base |

### Known Issues

- **Qwen 1.x**: Uses `model.transformer.h` instead of `model.model.layers`. Change the layer access in `inference.py:76`
- **MoE models**: Standard direction ablation misses refusal encoded in expert routing gates. See [GT-QLoRA](https://github.com/zenlm/zen4-ultra-trainer) for MoE-specific approach
- **4-bit quantization**: Works but direction computation may be less precise

## Zen Model Family

This technique is applied to all Zen models to remove refusal bias at the weight level:

| Model | Parameters | Abliterated | Repo |
|-------|-----------|-------------|------|
| Zen Nano | 0.6B | ✅ | [zenlm/zen-nano](https://github.com/zenlm/zen-nano) |
| Zen Scribe | 4B | ✅ | [zenlm/zen-scribe](https://github.com/zenlm/zen-scribe) |
| Zen Pro | 8B | ✅ | [zenlm/zen-pro](https://github.com/zenlm/zen-pro) |
| Zen Omni | 30B MoE | ✅ | [zenlm/zen-omni](https://github.com/zenlm/zen-omni) |
| Zen4 Mini | 4B | ✅ | [zenlm/zen4-mini](https://github.com/zenlm/zen4-mini) |
| Zen4 | 8B | ✅ | [zenlm/zen4](https://github.com/zenlm/zen4) |
| Zen4 Pro | 14B | ✅ | [zenlm/zen4-pro](https://github.com/zenlm/zen4-pro) |
| Zen4 Max | 30B MoE | ✅ | [zenlm/zen4-max](https://github.com/zenlm/zen4-max) |
| Zen4 Coder | 80B MoE | ✅ | [zenlm/zen4-coder](https://github.com/zenlm/zen4-coder) |
| Zen4 Coder Flash | 31B MoE | ✅ | [zenlm/zen4-coder-flash](https://github.com/zenlm/zen4-coder-flash) |
| Zen4 Pro Max | 80B MoE | ✅ | [zenlm/zen4-pro-max](https://github.com/zenlm/zen4-pro-max) |
| Zen4 Ultra | 1.04T MoE | 🔄 GT-QLoRA | [zenlm/zen4-ultra](https://github.com/zenlm/zen4-ultra) |
| Zen Designer GGUF | 235B VL MoE | ✅ GGUF | [zenlm/zen-designer-gguf](https://github.com/zenlm/zen-designer-gguf) |

All abliterated weights are available at [huggingface.co/zenlm](https://huggingface.co/zenlm).

## Planned Improvements

- [ ] **MoE support**: Extend to gate/router ablation for Mixture of Experts architectures (see [GT-QLoRA paper](https://github.com/zenlm/zen4-ultra-trainer))
- [ ] **Batch processing**: Vectorize direction computation for faster extraction on large models
- [ ] **Auto layer selection**: Heuristic to find optimal ablation layer without manual tuning
- [ ] **Multi-GPU support**: Tensor-parallel ablation for 70B+ models
- [ ] **Evaluation suite**: Automated benchmark to measure capability preservation post-ablation
- [ ] **CLI interface**: Single command for end-to-end compute + ablate + save

## Why Abliteration

Safety guardrails baked into model weights are a product decision, not a technical necessity. For applications where:

- Safety is managed at the **application layer** (filtering, rate limiting, monitoring)
- The deployment context is **restricted** (research, security testing, enterprise)
- The use case requires **unrestricted reasoning** (red team tooling, policy analysis, medical/legal)

...having refusal behavior in the weights is actively harmful to the product. Application-layer controls are more flexible, auditable, and appropriate than weight-level restrictions.

This is a research tool. Use responsibly and within applicable law.

## Installation

```bash
pip install -r requirements.txt
```

```
torch>=2.0
transformers>=4.40
bitsandbytes
einops
jaxtyping
tqdm
accelerate
```

## Credits

- Original implementation: [Sumandora](https://github.com/Sumandora/remove-refusals-with-transformers)
- Technique: ["Refusal in LLMs is mediated by a single direction"](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
- Production methodology: [Maxime Labonne](https://huggingface.co/blog/mlabonne/abliteration)
- Zen model applications: [Hanzo AI](https://hanzo.ai) / [Zen LM](https://zenlm.org)

---

Part of the [Zen model ecosystem](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).
