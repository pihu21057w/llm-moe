from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .model import DecoderOnlyTransformer
from .tokenizer import GPT4Tokenizer
from .tooling import parse_tool_call


@torch.no_grad()
def generate(model: DecoderOnlyTransformer, tokenizer: GPT4Tokenizer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 50) -> str:
    model.eval()
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    for _ in range(max_new_tokens):
        outputs = model(input_ids[:, -model.config.max_seq_len :])
        logits = outputs["logits"][:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, indices, values)
            logits = filtered
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model = DecoderOnlyTransformer(config)
    load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    tokenizer = GPT4Tokenizer(encoding_name="cl100k_base", special_tokens=config.tool_call_tokens)
    output = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(output)
    tool_call = parse_tool_call(output)
    if tool_call is not None:
        print(f"Detected tool call: {tool_call.name}")
