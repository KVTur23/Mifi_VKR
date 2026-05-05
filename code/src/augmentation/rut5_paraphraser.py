"""
ruT5 paraphraser wrapper with tokenizer-aware chunking.

Used by augmentation stage 2 to paraphrase full emails with
fyaronskiy/ruT5-large-paraphraser without silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from src.augmentation.text_chunking import chunk_text, join_chunks


DEFAULT_MODEL = "fyaronskiy/ruT5-large-paraphraser"


@dataclass
class RuT5ParaphraserConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 8
    chunk_max_tokens: int = 420
    output_length_factor: float = 1.5
    output_max_tokens: int = 512
    num_beams: int = 5
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    encoder_no_repeat_ngram_size: int = 3
    no_repeat_ngram_size: int = 3


class RuT5Paraphraser:
    def __init__(self, cfg: RuT5ParaphraserConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ruT5] Загружаю парафразер: {cfg.model_name} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[ruT5] Модель загружена")

    @classmethod
    def from_pipeline_config(cls, pipeline_cfg: Any | None) -> "RuT5Paraphraser":
        raw = {}
        if pipeline_cfg is not None:
            raw = dict(pipeline_cfg.stage2.get("paraphraser", {}))
        cfg = RuT5ParaphraserConfig(
            model_name=raw.get("model", DEFAULT_MODEL),
            batch_size=int(raw.get("batch_size", RuT5ParaphraserConfig.batch_size)),
            chunk_max_tokens=int(raw.get("chunk_max_tokens", RuT5ParaphraserConfig.chunk_max_tokens)),
            output_length_factor=float(raw.get("output_length_factor", RuT5ParaphraserConfig.output_length_factor)),
            output_max_tokens=int(raw.get("output_max_tokens", RuT5ParaphraserConfig.output_max_tokens)),
            num_beams=int(raw.get("num_beams", RuT5ParaphraserConfig.num_beams)),
            do_sample=bool(raw.get("do_sample", RuT5ParaphraserConfig.do_sample)),
            temperature=float(raw.get("temperature", RuT5ParaphraserConfig.temperature)),
            top_p=float(raw.get("top_p", RuT5ParaphraserConfig.top_p)),
            repetition_penalty=float(raw.get("repetition_penalty", RuT5ParaphraserConfig.repetition_penalty)),
            encoder_no_repeat_ngram_size=int(raw.get(
                "encoder_no_repeat_ngram_size",
                RuT5ParaphraserConfig.encoder_no_repeat_ngram_size,
            )),
            no_repeat_ngram_size=int(raw.get(
                "no_repeat_ngram_size",
                RuT5ParaphraserConfig.no_repeat_ngram_size,
            )),
        )
        return cls(cfg)

    def paraphrase_texts(
        self,
        texts: list[str],
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> list[str]:
        if not texts:
            return []

        flat_chunks: list[str] = []
        flat_input_tokens: list[int] = []
        owners: list[int] = []
        forced_splits = 0
        multi_chunk_docs = 0

        for doc_idx, text in enumerate(texts):
            chunks = chunk_text(text, self.tokenizer, self.cfg.chunk_max_tokens)
            if len(chunks) > 1:
                multi_chunk_docs += 1
            forced_splits += sum(1 for c in chunks if c.forced_split)
            for chunk in chunks:
                flat_chunks.append(chunk.text)
                flat_input_tokens.append(chunk.token_count)
                owners.append(doc_idx)

        print(
            f"[ruT5] Писем: {len(texts)}, чанков: {len(flat_chunks)}, "
            f"multi-chunk: {multi_chunk_docs}, forced splits: {forced_splits}, "
            f"temperature={temperature if temperature is not None else self.cfg.temperature:.2f}"
        )

        generated_chunks = self._generate_chunks(
            flat_chunks,
            flat_input_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        grouped: list[list[str]] = [[] for _ in texts]
        for owner, generated in zip(owners, generated_chunks):
            grouped[owner].append(generated)

        return [join_chunks(parts) for parts in grouped]

    def unload(self) -> None:
        self.model.cpu()
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ruT5] Модель выгружена")

    def _generate_chunks(
        self,
        chunks: list[str],
        input_token_counts: list[int],
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> list[str]:
        outputs: list[str] = []
        generation_temperature = self.cfg.temperature if temperature is None else temperature
        generation_top_p = self.cfg.top_p if top_p is None else top_p
        generation_do_sample = self.cfg.do_sample if do_sample is None else do_sample

        for start in tqdm(
            range(0, len(chunks), self.cfg.batch_size),
            desc="[ruT5] paraphrase",
            leave=False,
        ):
            batch = chunks[start:start + self.cfg.batch_size]
            batch_counts = input_token_counts[start:start + self.cfg.batch_size]
            max_length = min(
                int(max(batch_counts) * self.cfg.output_length_factor) + 8,
                self.cfg.output_max_tokens,
            )
            max_length = max(max_length, 32)
            min_length = max(int(min(batch_counts) * 0.7), 16)
            min_length = min(min_length, max_length - 1)

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=self.cfg.num_beams,
                    do_sample=generation_do_sample,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
                    repetition_penalty=self.cfg.repetition_penalty,
                    encoder_no_repeat_ngram_size=self.cfg.encoder_no_repeat_ngram_size,
                    no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
                )

            outputs.extend(
                self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )

        return [text.strip() for text in outputs]
