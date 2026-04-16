from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, IterableDataset, random_split

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

from .tokenizer import GPT4Tokenizer


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_documents(data_path: str, max_documents: int | None = None) -> list[str]:
    path = Path(data_path)
    documents: list[str] = []
    candidates: list[Path]
    if path.is_dir():
        candidates = sorted([candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in {".txt", ".jsonl", ".json"}])
    else:
        candidates = [path]

    for candidate in candidates:
        if candidate.suffix.lower() == ".txt":
            documents.extend(candidate.read_text(encoding="utf-8").splitlines())
        elif candidate.suffix.lower() == ".jsonl":
            for row in _read_jsonl(candidate):
                if "messages" in row and isinstance(row["messages"], list):
                    pieces = []
                    for message in row["messages"]:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        pieces.append(f"<{role}>{content}</{role}>")
                    documents.append("\n".join(pieces))
                else:
                    documents.append(str(row.get("text", "")))
        elif candidate.suffix.lower() == ".json":
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        documents.append(str(item.get("text", "")))
                    else:
                        documents.append(str(item))
            else:
                documents.append(str(raw))
        if max_documents is not None and len(documents) >= max_documents:
            break

    return [doc for doc in documents if doc.strip()]


def _detect_text_columns(columns: list[str], rows: list[dict], limit: int = 0) -> list[str]:
    detected: list[str] = []
    for column in columns:
        for row in rows:
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                detected.append(column)
                if limit > 0 and len(detected) >= limit:
                    return detected
                break
    return detected


def _combine_row_text(row: dict, text_columns: list[str]) -> str:
    parts: list[str] = []
    for column in text_columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _read_hf_documents(
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_columns: tuple[str, ...],
    max_text_columns: int,
    max_documents: int | None,
    streaming: bool,
) -> list[str]:
    if load_dataset is None:
        raise ImportError("datasets package is not installed. Please install 'datasets' to use Hugging Face data sources.")
    if not dataset_name:
        raise ValueError("hf_dataset_name is required when data_source='hf'")

    ds = load_dataset(dataset_name, dataset_config or None, split=split, streaming=streaming)
    if max_documents is not None:
        if streaming:
            documents: list[str] = []
            for index, row in enumerate(ds):
                if index >= max_documents:
                    break
                combined = _combine_row_text(row, list(text_columns))
                if combined:
                    documents.append(combined)
            return documents
        ds = ds.select(range(min(max_documents, len(ds))))

    if text_columns:
        columns = list(text_columns[:max_text_columns]) if max_text_columns > 0 else list(text_columns)
    else:
        sample_rows: list[dict] = []
        if streaming:
            for index, row in enumerate(ds):
                if index >= 64:
                    break
                sample_rows.append(row)
        else:
            sample_rows = [ds[index] for index in range(min(len(ds), 64))]
        if hasattr(ds, "column_names"):
            available_columns = list(ds.column_names)
        elif sample_rows:
            available_columns = list(sample_rows[0].keys())
        else:
            available_columns = []
        columns = _detect_text_columns(available_columns, sample_rows, limit=max_text_columns)
        if not columns:
            raise ValueError("No usable text columns found in Hugging Face dataset")

    documents: list[str] = []
    for row in ds:
        combined = _combine_row_text(row, columns)
        if combined:
            documents.append(combined)
        if max_documents is not None and len(documents) >= max_documents:
            break
    return documents


class StreamingTokenBlockDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str,
        split: str,
        text_columns: tuple[str, ...],
        max_text_columns: int,
        tokenizer: GPT4Tokenizer,
        block_size: int,
        validation_split: float,
        mode: str,
        max_documents: int | None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_columns = text_columns
        self.max_text_columns = max_text_columns
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.validation_split = validation_split
        self.mode = mode
        self.max_documents = max_documents
        self.newline_tokens = tokenizer.encode("\n")
        self.validation_stride = max(1, round(1.0 / max(validation_split, 1e-6))) if validation_split > 0 else 0

    def __iter__(self):
        if load_dataset is None:
            raise ImportError("datasets package is not installed. Please install 'datasets' to use Hugging Face data sources.")
        if not self.dataset_name:
            raise ValueError("hf_dataset_name is required when data_source='hf'")

        ds = load_dataset(self.dataset_name, self.dataset_config or None, split=self.split, streaming=True)
        columns = list(self.text_columns[: self.max_text_columns]) if self.max_text_columns > 0 and self.text_columns else list(self.text_columns)
        if not columns:
            sample_rows: list[dict] = []
            for index, row in enumerate(ds):
                if index >= 64:
                    break
                sample_rows.append(row)
            available_columns = list(sample_rows[0].keys()) if sample_rows else []
            columns = _detect_text_columns(available_columns, sample_rows, limit=self.max_text_columns)
            if not columns:
                raise ValueError("No usable text columns found in Hugging Face dataset")
            ds = load_dataset(self.dataset_name, self.dataset_config or None, split=self.split, streaming=True)

        buffer: list[int] = []
        document_count = 0
        block_index = 0
        for row in ds:
            if self.max_documents is not None and document_count >= self.max_documents:
                break
            document = _combine_row_text(row, columns)
            if not document:
                continue
            document_count += 1
            buffer.extend(self.tokenizer.encode(document))
            buffer.extend(self.newline_tokens)
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size :]
                is_validation = self.validation_stride > 0 and (block_index % self.validation_stride == 0)
                should_yield = (self.mode == "validation" and is_validation) or (self.mode == "train" and not is_validation)
                if should_yield:
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
                block_index += 1


class TokenBlockDataset(Dataset):
    def __init__(self, documents: Iterable[str], tokenizer: GPT4Tokenizer, block_size: int) -> None:
        self.block_size = block_size
        token_ids: list[int] = []
        newline_tokens = tokenizer.encode("\n")
        for document in documents:
            token_ids.extend(tokenizer.encode(document))
            token_ids.extend(newline_tokens)
        if len(token_ids) <= block_size + 1:
            raise ValueError("Dataset is too small for the requested block size")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, (self.tokens.numel() - 1) // self.block_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        if chunk.numel() < self.block_size + 1:
            padding = torch.full((self.block_size + 1 - chunk.numel(),), fill_value=self.tokens[-1].item(), dtype=torch.long)
            chunk = torch.cat([chunk, padding], dim=0)
        return chunk[:-1], chunk[1:]


@dataclass(slots=True)
class DatasetBundle:
    train: Dataset
    validation: Dataset


def build_datasets(
    data_path: str,
    tokenizer: GPT4Tokenizer,
    sequence_length: int,
    validation_split: float = 0.005,
    seed: int = 42,
    max_documents: int | None = None,
    data_source: str = "local",
    hf_streaming: bool = False,
    hf_dataset_name: str = "",
    hf_dataset_config: str = "",
    hf_split: str = "train",
    hf_text_columns: tuple[str, ...] = (),
    hf_max_text_columns: int = 0,
) -> DatasetBundle:
    if data_source == "hf":
        if hf_streaming:
            return DatasetBundle(
                train=StreamingTokenBlockDataset(
                    dataset_name=hf_dataset_name,
                    dataset_config=hf_dataset_config,
                    split=hf_split,
                    text_columns=hf_text_columns,
                    tokenizer=tokenizer,
                    block_size=sequence_length,
                    validation_split=validation_split,
                    mode="train",
                    max_text_columns=hf_max_text_columns,
                    max_documents=max_documents,
                ),
                validation=StreamingTokenBlockDataset(
                    dataset_name=hf_dataset_name,
                    dataset_config=hf_dataset_config,
                    split=hf_split,
                    text_columns=hf_text_columns,
                    tokenizer=tokenizer,
                    block_size=sequence_length,
                    validation_split=validation_split,
                    mode="validation",
                    max_text_columns=hf_max_text_columns,
                    max_documents=max_documents,
                ),
            )
        documents = _read_hf_documents(
            dataset_name=hf_dataset_name,
            dataset_config=hf_dataset_config,
            split=hf_split,
            text_columns=hf_text_columns,
            max_text_columns=hf_max_text_columns,
            max_documents=max_documents,
            streaming=False,
        )
    else:
        documents = _read_documents(data_path, max_documents=max_documents)
    dataset = TokenBlockDataset(documents, tokenizer=tokenizer, block_size=sequence_length)
    if len(dataset) < 2:
        train_size = len(dataset)
        validation_size = 0
    else:
        validation_size = max(1, int(len(dataset) * validation_split))
        validation_size = min(validation_size, len(dataset) - 1)
        train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    return DatasetBundle(train=train_dataset, validation=validation_dataset)
