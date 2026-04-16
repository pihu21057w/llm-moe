from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint, write_config
from .config import DataConfig, ModelConfig, TrainConfig
from .data import build_datasets
from .model import DecoderOnlyTransformer
from .monitoring import TrainingMonitor
from .tokenizer import GPT4Tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_schedule(step: int, train_config: TrainConfig) -> float:
    if step < train_config.optim.warmup_steps:
        return train_config.optim.lr * (step + 1) / max(1, train_config.optim.warmup_steps)
    progress = (step - train_config.optim.warmup_steps) / max(1, train_config.max_steps - train_config.optim.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return train_config.optim.lr * (train_config.optim.min_lr_ratio + (1.0 - train_config.optim.min_lr_ratio) * cosine)


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def evaluate(model: DecoderOnlyTransformer, dataloader: DataLoader, device: torch.device, dtype: torch.dtype, max_batches: int) -> dict[str, float]:
    model.eval()
    losses = []
    aux_losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            inputs, targets = (tensor.to(device, non_blocking=True) for tensor in batch)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs["logits"].view(-1, outputs["logits"].size(-1)), targets.view(-1))
            losses.append(float(loss.item()))
            aux_losses.append(float(outputs["moe_aux_loss"].item()))
    model.train()
    return {"eval/loss": float(np.mean(losses)) if losses else 0.0, "eval/moe_aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0}


def train_loop(model: DecoderOnlyTransformer, train_loader: DataLoader, validation_loader: DataLoader, tokenizer: GPT4Tokenizer, model_config: ModelConfig, train_config: TrainConfig, device: torch.device) -> None:
    dtype = get_dtype(train_config.dtype)
    model = model.to(device)
    if train_config.compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.optim.lr, betas=train_config.optim.betas, weight_decay=train_config.optim.weight_decay)
    scheduler = None
    global_step = 0
    if train_config.resume:
        checkpoint = load_checkpoint(train_config.resume, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        global_step = int(checkpoint.get("step", 0)) + 1

    monitor = TrainingMonitor(
        train_config.output_path(),
        tensorboard=train_config.monitor_tensorboard,
        csv_logging=train_config.monitor_csv,
        wandb_enabled=train_config.monitor_wandb,
        wandb_project=train_config.wandb_project,
        wandb_run_name=train_config.wandb_run_name,
    )
    steps_per_checkpoint = max(1, train_config.save_every)
    output_dir = train_config.output_path()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir / "config.json", {"model": model_config.to_dict(), "train": asdict(train_config)})

    model.train()
    start_time = perf_counter()
    progress = tqdm(total=train_config.max_steps, initial=global_step, desc="training")
    train_iterator = iter(train_loader)
    while global_step < train_config.max_steps:
        step_start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss_value = 0.0
        total_lm_value = 0.0
        total_aux_value = 0.0
        for accumulation_step in range(train_config.grad_accum_steps):
            try:
                inputs, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                inputs, targets = next(train_iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(inputs)
                lm_loss = torch.nn.functional.cross_entropy(outputs["logits"].view(-1, outputs["logits"].size(-1)), targets.view(-1))
                loss = lm_loss + 0.01 * outputs["moe_aux_loss"]
                loss = loss / train_config.grad_accum_steps

            loss.backward()
            total_loss_value += float(loss.item())
            total_lm_value += float(lm_loss.item())
            total_aux_value += float(outputs["moe_aux_loss"].item())

        if train_config.optim.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.optim.grad_clip_norm)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        lr = make_lr_schedule(global_step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        step_elapsed = perf_counter() - step_start
        tokens_per_second = (train_config.micro_batch_size * train_config.grad_accum_steps * inputs.numel()) / max(step_elapsed, 1e-6) if train_config.report_tokens_per_second else 0.0
        metrics = {
            "train/loss": total_loss_value / train_config.grad_accum_steps,
            "train/lm_loss": total_lm_value / train_config.grad_accum_steps,
            "train/moe_aux_loss": total_aux_value / train_config.grad_accum_steps,
            "train/lr": lr,
            "train/grad_norm": float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "train/tokens_per_second": float(tokens_per_second),
        }

        if global_step % train_config.log_every == 0:
            monitor.log(global_step, metrics)
            monitor.report_step(global_step, metrics)

        if global_step % train_config.eval_every == 0 and validation_loader is not None:
            eval_metrics = evaluate(model, validation_loader, device, dtype, train_config.eval_batches)
            monitor.log(global_step, eval_metrics)
            monitor.report_step(global_step, eval_metrics)

        if global_step % steps_per_checkpoint == 0 or global_step == train_config.max_steps - 1:
            save_checkpoint(
                checkpoint_dir / f"step-{global_step:07d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                config=model_config.to_dict(),
                extra_state={"train_config": asdict(train_config)},
                save_optimizer_state=train_config.save_optimizer_state,
            )
            if train_config.save_last:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    config=model_config.to_dict(),
                    extra_state={"train_config": asdict(train_config)},
                    save_optimizer_state=train_config.save_optimizer_state,
                )

        global_step += 1
        progress.update(1)

    progress.close()
    monitor.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a decoder-only transformer")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="runs/default")
    parser.add_argument("--preset", default="1b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "flash", "sdpa", "linear"])
    parser.add_argument("--d-model", type=int, default=1536)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=6144)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-scaling", default="linear", choices=["none", "linear", "ntk"])
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)
    parser.add_argument("--moe-every-n-layers", type=int, default=4)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    model_config = ModelConfig.preset_1b() if args.preset == "1b" else ModelConfig()
    model_config.d_model = args.d_model
    model_config.n_layers = args.n_layers
    model_config.n_heads = args.n_heads
    model_config.n_kv_heads = args.n_kv_heads
    model_config.d_ff = args.d_ff
    model_config.dropout = args.dropout
    model_config.max_seq_len = args.sequence_length
    model_config.attention_backend = args.attention_backend
    model_config.rope_scaling = args.rope_scaling
    model_config.rope_scaling_factor = args.rope_scaling_factor
    model_config.moe_every_n_layers = args.moe_every_n_layers
    model_config.moe_num_experts = args.moe_num_experts
    model_config.moe_top_k = args.moe_top_k
    model_config.use_checkpointing = args.activation_checkpointing
    data_config = DataConfig(data_path=args.data, tokenizer_name=args.tokenizer, sequence_length=args.sequence_length)
    train_config = TrainConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        dtype=args.dtype,
        resume=args.resume,
        monitor_wandb=args.wandb,
        use_activation_checkpointing=args.activation_checkpointing,
    )
    train_config.optim.lr = args.lr
    train_config.optim.weight_decay = args.weight_decay
    train_config.optim.warmup_steps = args.warmup_steps
    train_config.optim.grad_clip_norm = args.grad_clip_norm

    tokenizer = GPT4Tokenizer(encoding_name=data_config.tokenizer_name, special_tokens=model_config.tool_call_tokens)
    model_config.vocab_size = tokenizer.vocab_size
    datasets = build_datasets(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        sequence_length=data_config.sequence_length,
        validation_split=data_config.validation_split,
        seed=data_config.seed,
        max_documents=data_config.max_documents,
    )
    train_loader = DataLoader(datasets.train, batch_size=train_config.micro_batch_size, shuffle=True, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory, drop_last=True)
    validation_loader = DataLoader(datasets.validation, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory, drop_last=False)

    device = torch.device(train_config.device if torch.cuda.is_available() or train_config.device == "cpu" else "cpu")
    model = DecoderOnlyTransformer(model_config)
    train_loop(model, train_loader, validation_loader, tokenizer, model_config, train_config, device)


if __name__ == "__main__":
    main()
