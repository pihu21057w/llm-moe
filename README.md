# llmcore

Production-oriented decoder-only transformer training stack for Colab and Modal.

## What This Project Includes

- Decoder-only transformer with grouped-query attention
- MoE feed-forward layers (configurable frequency, experts, and top-k routing)
- RoPE with scaling modes
- Attention backends: auto, flash, sdpa, linear
- Reasoning-style refinement steps inside each decoder block
- GPT-4-class tokenization using tiktoken (cl100k_base)
- Local or Hugging Face dataset ingestion with multi-column text fusion
- CSV, TensorBoard, and optional W&B logging
- Periodic checkpoints + latest checkpoint + resume support
- Tool-calling prompt format and parser utilities

## Install

### Local / VM / Modal

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Colab

```python
!git clone https://github.com/pihu21057w/llm-moe.git
%cd llm-moe
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
```

Flash attention is optional. If unavailable, training falls back to PyTorch SDPA automatically.

## Entrypoints

- Training CLI: `llmcore-train`
- Generation CLI: `llmcore-generate`

## Quickstart: Train On Local Files

```bash
llmcore-train \
  --data /path/to/dataset_or_folder \
  --data-source local \
  --output-dir runs/local-exp \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --grad-accum-steps 16 \
  --max-steps 2000 \
  --attention-backend auto \
  --activation-checkpointing
```

Supported local formats:

- .txt
- .json
- .jsonl

JSONL supports either a text field or messages arrays.

## Quickstart: Train On Hugging Face Dataset

Example for pritamdeb68/Small-LM-Pretraining:

```bash
llmcore-train \
  --data-source hf \
  --hf-dataset pritamdeb68/Small-LM-Pretraining \
  --hf-split train \
  --hf-text-columns text,title,summary,content \
  --output-dir runs/hf-small-lm \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --grad-accum-steps 16 \
  --max-steps 2000 \
  --attention-backend auto \
  --reasoning-steps 3 \
  --reasoning-loss-weight 0.03 \
  --activation-checkpointing
```

Notes:

- If hf-text-columns is omitted, the loader auto-detects non-empty string columns.
- Multiple text columns are concatenated per row with newline separators.

## Reasoning Architecture

This project keeps a standard decoder-only causal LM objective while adding iterative reasoning refinement steps in each block.

Each layer performs:

1. Self-attention
2. FFN or MoE FFN
3. N reasoning refinement steps

Reasoning controls:

- --reasoning-steps: number of refinement steps per block
- --reasoning-loss-weight: auxiliary loss coefficient
- --reasoning-residual-scale: scale of each reasoning residual update
- --reasoning-hidden-mult: hidden size multiplier for reasoning FFN
- --reasoning-no-share-parameters: use separate reasoning cell per block

## Key Training Flags

Data:

- --data
- --data-source local|hf
- --hf-dataset
- --hf-config
- --hf-split
- --hf-text-columns
- --validation-split
- --max-documents

Model:

- --d-model
- --n-layers
- --n-heads
- --n-kv-heads
- --d-ff
- --dropout
- --attention-backend auto|flash|sdpa|linear
- --rope-scaling none|linear|ntk
- --rope-scaling-factor
- --moe-every-n-layers
- --moe-num-experts
- --moe-top-k
- --reasoning-steps
- --reasoning-residual-scale
- --reasoning-hidden-mult
- --reasoning-no-share-parameters

Optimization and runtime:

- --lr
- --weight-decay
- --warmup-steps
- --grad-clip-norm
- --dtype bfloat16|float16|float32
- --compile-model
- --activation-checkpointing
- --micro-batch-size
- --grad-accum-steps
- --max-steps

Checkpointing and logging:

- --save-every
- --eval-every
- --log-every
- --resume
- --wandb

## Monitoring And Artifacts

Outputs are written under output-dir:

- config.json: serialized run configuration
- metrics.csv: per-step scalar logs
- tensorboard/: TensorBoard events
- checkpoints/step-XXXXXXX.pt: periodic checkpoints
- checkpoints/latest.pt: rolling latest checkpoint

Resume training:

```bash
llmcore-train \
  --data-source hf \
  --hf-dataset pritamdeb68/Small-LM-Pretraining \
  --output-dir runs/hf-small-lm \
  --resume runs/hf-small-lm/checkpoints/latest.pt
```

## Inference

Generate from a checkpoint:

```bash
llmcore-generate \
  --checkpoint runs/hf-small-lm/checkpoints/latest.pt \
  --prompt "<system>You are helpful.</system><user>Give 3 tips for model training.</user><assistant>" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 50
```

## Tool Calling Interface

Tool-call tags are included in the tokenizer special tokens by default.

Tags:

- <tool_call>...</tool_call>
- <tool_result>...</tool_result>

Helper APIs:

- format_tool_call(name, arguments)
- parse_tool_call(text)
- ToolRegistry / ToolSpec for schema and execution wiring

Minimal example:

```python
from llmcore.tooling import ToolRegistry, ToolSpec, format_tool_call

registry = ToolRegistry()
registry.register(
    ToolSpec(
        name="search_docs",
        description="Search internal docs",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda args: {"results": [f"query={args['query']}"]},
    )
)

prompt_tools = registry.as_prompt_block()
tool_request = format_tool_call("search_docs", {"query": "rope scaling"})
```

## Colab 15 GB Preset

Suggested starter values:

- --sequence-length 1024
- --micro-batch-size 1
- --grad-accum-steps 16 (or higher)
- --activation-checkpointing
- --dtype bfloat16

For 24 to 48 GB GPUs, scale sequence length and micro-batch-size gradually.

## Troubleshooting

- Out-of-memory: lower sequence length, raise grad accumulation, enable activation checkpointing.
- Slow throughput: try attention backend auto or flash and tune sequence length.
- HF load errors: verify dataset name, split, and text columns.
- Diverging loss: reduce learning rate or increase warmup steps.
