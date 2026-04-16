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
  --hf-streaming \
  --hf-max-text-columns 4 \
  --max-documents 50000 \
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
- Use --hf-streaming for Hugging Face datasets that should be read lazily instead of loaded into RAM.
- Use --hf-max-text-columns to cap the number of columns consumed when the dataset schema is very wide.
- For very wide datasets, prefer an explicit fixed column list in --hf-text-columns.
- If you only want a fixed amount of data, use --max-documents to cap the number of rows/documents read.

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

## Configuration API

The project exposes typed dataclass configs in `llmcore.config`:

- `ModelConfig`: architecture settings
- `DataConfig`: data source and tokenization settings
- `TrainConfig`: runtime, logging, checkpoint, and optimizer settings
- `OptimConfig`: nested inside `TrainConfig.optim`

### Python config example

```python
from llmcore.config import ModelConfig, DataConfig, TrainConfig

model_cfg = ModelConfig(
  d_model=1536,
  n_layers=24,
  n_heads=16,
  n_kv_heads=4,
  d_ff=6144,
  max_seq_len=1024,
  attention_backend="auto",
  rope_scaling="linear",
  rope_scaling_factor=1.0,
  moe_every_n_layers=4,
  moe_num_experts=4,
  moe_top_k=2,
  reasoning_steps=3,
  reasoning_share_parameters=True,
)

data_cfg = DataConfig(
  data_source="hf",
  hf_dataset_name="pritamdeb68/Small-LM-Pretraining",
  hf_split="train",
  hf_text_columns=("text", "title", "summary", "content"),
  hf_streaming=True,
  hf_max_text_columns=4,
  sequence_length=1024,
  validation_split=0.005,
)

train_cfg = TrainConfig(
  output_dir="runs/hf-small-lm",
  max_steps=2000,
  micro_batch_size=1,
  grad_accum_steps=16,
  dtype="bfloat16",
  save_every=100,
  eval_every=100,
  log_every=1,
  use_activation_checkpointing=True,
  reasoning_loss_weight=0.03,
)

train_cfg.optim.lr = 3e-4
train_cfg.optim.weight_decay = 0.1
train_cfg.optim.warmup_steps = 200
train_cfg.optim.grad_clip_norm = 1.0
```

### Config field groups

ModelConfig includes:

- Core shape: `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_ff`
- Attention: `attention_backend`, `rope_scaling`, `rope_scaling_factor`, `rope_theta`
- MoE: `moe_every_n_layers`, `moe_num_experts`, `moe_top_k`, `moe_hidden_mult`
- Reasoning: `reasoning_steps`, `reasoning_share_parameters`, `reasoning_residual_scale`, `reasoning_hidden_mult`
- Misc: `dropout`, `tie_embeddings`, `use_rms_norm`, `use_checkpointing`

DataConfig includes:

- Local data: `data_path`, `data_source="local"`
- Hugging Face: `data_source="hf"`, `hf_dataset_name`, `hf_dataset_config`, `hf_split`, `hf_text_columns`
- Hugging Face streaming: `hf_streaming`, `hf_max_text_columns`
- Row/document cap: `max_documents`
- Packing: `sequence_length`, `validation_split`

TrainConfig includes:

- Loop control: `max_steps`, `micro_batch_size`, `grad_accum_steps`
- Runtime: `device`, `dtype`, `compile_model`, `num_workers`, `pin_memory`
- Monitoring: `monitor_tensorboard`, `monitor_csv`, `monitor_wandb`
- Checkpoints: `save_every`, `save_last`, `save_optimizer_state`, `resume`
- Reasoning objective: `reasoning_loss_weight`
- Optimizer block: `train_cfg.optim` (`lr`, `betas`, `weight_decay`, `warmup_steps`, `grad_clip_norm`, `min_lr_ratio`)

## Key Training Flags

Data:

- --data
- --data-source local|hf
- --hf-dataset
- --hf-config
- --hf-split
- --hf-text-columns
- --hf-streaming
- --hf-max-text-columns
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
