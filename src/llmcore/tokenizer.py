from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken


@dataclass(slots=True)
class TokenizerState:
    encoding_name: str
    special_tokens: tuple[str, ...]


class GPT4Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base", special_tokens: Iterable[str] | None = None) -> None:
        self.encoding_name = encoding_name
        self.base = tiktoken.get_encoding(encoding_name)
        self.special_tokens = tuple(special_tokens or ())
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        base_vocab_size = self.base.n_vocab
        for offset, token in enumerate(self.special_tokens):
            token_id = base_vocab_size + offset
            self.special_to_id[token] = token_id
            self.id_to_special[token_id] = token
        self.vocab_size = self.base.n_vocab + len(self.special_tokens)
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self._special_splitter = re.compile(f"({pattern})")
        else:
            self._special_splitter = None

    def encode(self, text: str) -> list[int]:
        if not self._special_splitter:
            return self.base.encode(text, allowed_special=set())
        tokens: list[int] = []
        for piece in self._special_splitter.split(text):
            if not piece:
                continue
            special_id = self.special_to_id.get(piece)
            if special_id is not None:
                tokens.append(special_id)
            else:
                tokens.extend(self.base.encode(piece, allowed_special=set()))
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces: list[str] = []
        buffer: list[int] = []
        for token_id in token_ids:
            special = self.id_to_special.get(int(token_id))
            if special is not None:
                if buffer:
                    pieces.append(self.base.decode(buffer))
                    buffer.clear()
                pieces.append(special)
            else:
                buffer.append(int(token_id))
        if buffer:
            pieces.append(self.base.decode(buffer))
        return "".join(pieces)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        text = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            text.append(f"<{role}>")
            text.append(content)
            text.append(f"</{role}>")
        return self.encode("".join(text))

    def state_dict(self) -> TokenizerState:
        return TokenizerState(encoding_name=self.encoding_name, special_tokens=self.special_tokens)
