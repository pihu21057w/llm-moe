from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


AttentionBackend = Literal["auto", "flash", "sdpa", "linear"]
RopeScalingType = Literal["none", "linear", "ntk"]


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 100_277
    max_seq_len: int = 2048
    d_model: int = 1536
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 6144
    dropout: float = 0.0
    attention_backend: AttentionBackend = "auto"
    rope_scaling: RopeScalingType = "linear"
    rope_scaling_factor: float = 1.0
    rope_theta: float = 10_000.0
    moe_every_n_layers: int = 4
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_hidden_mult: float = 1.0
    activation: str = "silu"
    tie_embeddings: bool = True
    use_bias: bool = False
    use_rms_norm: bool = True
    use_checkpointing: bool = False
    tool_call_tokens: tuple[str, ...] = (
        "<tool_call>",
        "</tool_call>",
        "<tool_result>",
        "</tool_result>",
        "<system>",
        "</system>",
        "<user>",
        "</user>",
        "<assistant>",
        "</assistant>",
    )

    @classmethod
    def preset_1b(cls) -> "ModelConfig":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DataConfig:
    data_path: str = ""
    tokenizer_name: str = "cl100k_base"
    sequence_length: int = 2048
    validation_split: float = 0.005
    max_documents: int | None = None
    seed: int = 42


@dataclass(slots=True)
class OptimConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "runs/default"
    max_steps: int = 50_000
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    log_every: int = 1
    save_every: int = 250
    eval_every: int = 500
    eval_batches: int = 20
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = False
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    monitor_tensorboard: bool = True
    monitor_csv: bool = True
    monitor_wandb: bool = False
    wandb_project: str = "llmcore"
    wandb_run_name: str = ""
    resume: str = ""
    save_last: bool = True
    save_optimizer_state: bool = True
    packed_examples: bool = True
    use_activation_checkpointing: bool = False
    grad_norm_log: bool = True
    report_tokens_per_second: bool = True
    optim: OptimConfig = field(default_factory=OptimConfig)

    def output_path(self) -> Path:
        return Path(self.output_dir)
