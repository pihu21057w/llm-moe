# llmcore

Decoder-only transformer training stack for Colab and Modal with:

- MoE feed-forward layers
- FlashAttention-style attention with automatic fallback
- Linear attention backend option
- RoPE with scaling support
- GPT-4-class tokenizer support through tiktoken
- Step-by-step logging to console, CSV, and TensorBoard
- Checkpoint saving and resume support
- Tool-calling friendly prompt format and parsing helpers
- Modular config surfaces for later extension
- Hugging Face dataset loading (including multi-column text fusion)
- Reasoning-style decoder blocks with iterative internal reasoning steps

## Colab Quickstart

Use these cells in a fresh Colab notebook after mounting your dataset or uploading text/JSONL files.

### 1. Clone and install

```python
!git clone https://github.com/pihu21057w/llm-moe.git
%cd llm-moe
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
```

If you want FlashAttention acceleration, install a build that matches your CUDA stack. The project automatically falls back to PyTorch SDPA if FlashAttention is unavailable.

### 2. Import the project

```python
import torch

from llmcore.config import ModelConfig, TrainConfig, DataConfig
from llmcore.model import DecoderOnlyTransformer
from llmcore.tokenizer import GPT4Tokenizer
from llmcore.data import build_datasets
from llmcore.train import train_loop, set_seed
from torch.utils.data import DataLoader
```

### 3. Configure model and training

```python
set_seed(42)

model_config = ModelConfig.preset_1b()
model_config.d_model = 1536
model_config.n_layers = 24
model_config.n_heads = 16
model_config.n_kv_heads = 4
model_config.d_ff = 6144
model_config.max_seq_len = 1024
model_config.attention_backend = "auto"
model_config.rope_scaling = "linear"
model_config.rope_scaling_factor = 1.0
model_config.moe_every_n_layers = 4
model_config.moe_num_experts = 4
model_config.moe_top_k = 2
model_config.use_checkpointing = True

train_config = TrainConfig(
	output_dir="runs/colab-1b",
	max_steps=2000,
	micro_batch_size=1,
	grad_accum_steps=16,
	save_every=100,
	eval_every=100,
	log_every=1,
	dtype="bfloat16",
	compile_model=False,
)

train_config.optim.lr = 3e-4
train_config.optim.weight_decay = 0.1
train_config.optim.warmup_steps = 200
train_config.optim.grad_clip_norm = 1.0
```

### 4. Prepare tokenizer and data

```python
data_config = DataConfig(
	data_path="/content/your-data-folder-or-file",
	tokenizer_name="cl100k_base",
	sequence_length=model_config.max_seq_len,
)

tokenizer = GPT4Tokenizer(
	encoding_name=data_config.tokenizer_name,
	special_tokens=model_config.tool_call_tokens,
)
model_config.vocab_size = tokenizer.vocab_size

datasets = build_datasets(
	data_path=data_config.data_path,
	tokenizer=tokenizer,
	sequence_length=data_config.sequence_length,
	validation_split=data_config.validation_split,
	seed=data_config.seed,
	max_documents=data_config.max_documents,
)

train_loader = DataLoader(
	datasets.train,
	batch_size=train_config.micro_batch_size,
	shuffle=True,
	num_workers=0,
	pin_memory=True,
	drop_last=True,
)

validation_loader = DataLoader(
	datasets.validation,
	batch_size=train_config.micro_batch_size,
	shuffle=False,
	num_workers=0,
	pin_memory=True,
	drop_last=False,
)
```

### 4b. Use Hugging Face dataset directly

For your dataset (`pritamdeb68/Small-LM-Pretraining`), use this CLI call in Colab:

```python
!llmcore-train \
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

If `--hf-text-columns` is omitted, the loader auto-detects string columns and concatenates them per row.

### 5. Create the model

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(model_config).to(device)
print(f"Parameters: {model.estimate_parameters():,}")
```

### 6. Train

```python
train_loop(
	model=model,
	train_loader=train_loader,
	validation_loader=validation_loader,
	tokenizer=tokenizer,
	model_config=model_config,
	train_config=train_config,
	device=device,
)
```

### 7. Resume from a checkpoint

```python
from llmcore.checkpoint import load_checkpoint

checkpoint_path = "runs/colab-1b/checkpoints/latest.pt"
load_checkpoint(checkpoint_path, model=model, map_location=device)
```

### 8. Inference interface

```python
from llmcore.inference import generate
from llmcore.tooling import format_tool_call, parse_tool_call, ToolRegistry, ToolSpec

prompt = "<system>You are a helpful assistant.</system><user>Write a haiku about transformers.</user><assistant>"
text = generate(
	model=model,
	tokenizer=tokenizer,
	prompt=prompt,
	max_new_tokens=128,
	temperature=0.8,
	top_k=50,
)
print(text)
```

### 9. Tool-calling example

```python
registry = ToolRegistry()
registry.register(
	ToolSpec(
		name="search_docs",
		description="Search internal documents by keyword",
		schema={
			"type": "object",
			"properties": {
				"query": {"type": "string"},
			},
			"required": ["query"],
		},
		handler=lambda args: {"results": [f"searched for {args['query']}"]},
	)
)

tool_block = registry.as_prompt_block()
tool_request = format_tool_call("search_docs", {"query": "transformer training"})
print(tool_block)
print(tool_request)
```

## Training Flags

Useful CLI flags:

- `--sequence-length 1024` for 15 GB GPUs
- `--micro-batch-size 1`
- `--grad-accum-steps 16`
- `--attention-backend auto|flash|sdpa|linear`
- `--resume runs/exp1/checkpoints/latest.pt`
- `--data-source local|hf`
- `--hf-dataset pritamdeb68/Small-LM-Pretraining`
- `--hf-text-columns text,title,summary,content`
- `--reasoning-steps 2`
- `--reasoning-loss-weight 0.02`

## Data Format

Supported sources are plain text files, directories of text files, and JSONL files with a `text` field. JSONL lines with `messages` are also supported and are serialized with role markers for tool-calling style fine-tuning.

## Notes

For a 15 GB Colab GPU, start with:

- `--micro-batch-size 1`
- `--grad-accum-steps 16` or higher
- `--sequence-length 1024` if memory is tight
- `--activation-checkpointing`

For Modal or larger GPUs, raise sequence length and micro-batch size gradually.
