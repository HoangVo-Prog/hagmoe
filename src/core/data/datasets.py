from __future__ import annotations

import json
import random
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


LABEL_KEYS = ("sentiment", "polarity", "label")
SENTENCE_KEYS = ("sentence", "text", "sentence_raw")
ASPECT_KEYS = ("aspect", "term", "aspect_raw", "target")
SAMPLE_LIST_KEYS = ("data", "samples", "instances")


def _first_key(sample: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in sample:
            return key
    return None


def _require_key(sample: dict, keys: Sequence[str], *, field: str, idx: int) -> str:
    key = _first_key(sample, keys)
    if key is None:
        tried = ", ".join(keys)
        available = ", ".join(sorted(sample.keys()))
        raise ValueError(
            f"Missing {field} field at sample index {idx}. Tried keys: [{tried}]. "
            f"Available keys: {available}"
        )
    return key


def _extract_sample_fields(sample: dict, *, idx: int) -> Tuple[str, str, object]:
    sent_key = _require_key(sample, SENTENCE_KEYS, field="sentence", idx=idx)
    aspect_key = _require_key(sample, ASPECT_KEYS, field="aspect", idx=idx)
    label_key = _require_key(sample, LABEL_KEYS, field="label", idx=idx)
    return sample[sent_key], sample[aspect_key], sample[label_key]


def _coerce_list_root(raw: object) -> List[dict]:
    if isinstance(raw, dict):
        for key in SAMPLE_LIST_KEYS:
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must be a list or contain a list under keys: data/samples/instances")
    out: List[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {i} is not an object: {type(item)}")
        out.append(item)
    return out


def _expand_aspect_lists(samples: List[dict]) -> List[dict]:
    out: List[dict] = []
    for i, item in enumerate(samples):
        if any(k in item for k in ASPECT_KEYS) and any(k in item for k in LABEL_KEYS):
            out.append(item)
            continue
        aspects = item.get("aspects")
        if aspects is None:
            aspects = item.get("aspect_terms")
        if aspects is None:
            aspects = item.get("targets")
        if aspects is None:
            out.append(item)
            continue
        if not isinstance(aspects, list):
            raise ValueError(f"'aspects' field must be a list at sample index {i}")
        if not aspects:
            raise ValueError(f"'aspects' list is empty at sample index {i}")
        sentence_key = _require_key(item, SENTENCE_KEYS, field="sentence", idx=i)
        sentence = item[sentence_key]
        for j, asp in enumerate(aspects):
            if not isinstance(asp, dict):
                raise ValueError(f"Aspect entry at sample index {i} item {j} is not an object")
            aspect_key = _require_key(asp, ASPECT_KEYS, field="aspect", idx=i)
            label_key = _require_key(asp, LABEL_KEYS, field="label", idx=i)
            merged = dict(item)
            merged.update({"aspect": asp[aspect_key], "label": asp[label_key], "sentence": sentence})
            out.append(merged)
    return out


def _load_samples(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = _coerce_list_root(raw)
    return _expand_aspect_lists(samples)

def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


_HYPHEN_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"
_HYPHEN_RE = re.compile(_HYPHEN_CLASS)


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _HYPHEN_RE.sub("-", s)
    s = s.strip(string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_wrapping_punct(s: str) -> str:
    return (s or "").strip(string.punctuation + "\"'")


def _find_aspect_char_span(sentence: str, term: str) -> Tuple[int, int]:
    if not sentence or not term:
        return -1, -1

    sentence_lower = sentence.lower()
    term_lower = term.lower()
    idx = sentence_lower.find(term_lower)
    if idx >= 0:
        return idx, idx + len(term)

    term_stripped = _strip_wrapping_punct(term)
    if term_stripped and term_stripped != term:
        idx = sentence_lower.find(term_stripped.lower())
        if idx >= 0:
            return idx, idx + len(term_stripped)

    term_norm = _normalize_text(term_stripped or term)
    if term_norm:
        tokens = [t for t in re.split(rf"[\s{_HYPHEN_CLASS}]+", term_norm) if t]
        if tokens:
            sep_pattern = rf"(?:\s+|[{_HYPHEN_CLASS}]+)"
            pattern = rf"\b{sep_pattern.join(map(re.escape, tokens))}\b"
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return match.start(), match.end()
            pattern = sep_pattern.join(map(re.escape, tokens))
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return match.start(), match.end()

    return -1, -1


def _token_span_from_offsets(
    offsets: Sequence[Tuple[int, int]], char_start: int, char_end: int
) -> Tuple[int, int]:
    tok_start = None
    tok_end = None
    for i, (start, end) in enumerate(offsets):
        if start is None or end is None:
            continue
        if start == 0 and end == 0:
            continue
        if end <= char_start:
            continue
        if start >= char_end:
            continue
        if tok_start is None:
            tok_start = i
        tok_end = i + 1
    if tok_start is None:
        return -1, -1
    return tok_start, tok_end


def _compute_aspect_span(
    *,
    tokenizer,
    term: str,
    sentence: str,
    max_len_sent: int,
    max_len_term: int,
):
    sentence_norm = _normalize_text(sentence)
    term_norm = _normalize_text(term)

    char_start, char_end = _find_aspect_char_span(sentence, term)
    if char_start < 0:
        sent_enc = tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=max_len_sent,
            return_tensors="pt",
        )
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        sent_ids = sent_enc["input_ids"].squeeze(0).tolist()
        sent_mask = sent_enc["attention_mask"].squeeze(0).tolist()
        valid_len = int(sum(sent_mask))
        sep_id = getattr(tokenizer, "sep_token_id", None)
        sep_idx = -1
        if sep_id is not None:
            try:
                sep_idx = sent_ids.index(sep_id)
            except ValueError:
                sep_idx = -1
        diag = {
            "sentence_norm": sentence_norm,
            "term_norm": term_norm,
            "content_ids": sent_ids[1:sep_idx] if sep_idx > 1 else [],
            "term_ids": [],
            "sent_tokens": tokenizer.convert_ids_to_tokens(sent_ids[1:sep_idx])
            if sep_idx > 1
            else [],
            "term_tokens": tokenizer.tokenize((term or "").lower()),
            "token_match_idx": -1,
            "valid_len": valid_len,
            "sep_idx": sep_idx,
        }
        return (
            sent_enc["input_ids"].squeeze(0),
            sent_enc["attention_mask"].squeeze(0),
            aspect_mask,
            0,
            0,
            "NOT_FOUND_RAW",
            False,
            diag,
        )

    full_enc = tokenizer(
        sentence,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    full_ids = full_enc["input_ids"].squeeze(0).tolist()
    offsets = full_enc["offset_mapping"].squeeze(0).tolist()

    tok_start, tok_end = _token_span_from_offsets(offsets, char_start, char_end)
    if tok_start < 0:
        sent_enc = tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=max_len_sent,
            return_tensors="pt",
        )
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        sent_ids = sent_enc["input_ids"].squeeze(0).tolist()
        sent_mask = sent_enc["attention_mask"].squeeze(0).tolist()
        valid_len = int(sum(sent_mask))
        sep_id = getattr(tokenizer, "sep_token_id", None)
        sep_idx = -1
        if sep_id is not None:
            try:
                sep_idx = sent_ids.index(sep_id)
            except ValueError:
                sep_idx = -1
        diag = {
            "sentence_norm": sentence_norm,
            "term_norm": term_norm,
            "content_ids": sent_ids[1:sep_idx] if sep_idx > 1 else [],
            "term_ids": [],
            "sent_tokens": tokenizer.convert_ids_to_tokens(sent_ids[1:sep_idx])
            if sep_idx > 1
            else [],
            "term_tokens": tokenizer.tokenize((term or "").lower()),
            "token_match_idx": -1,
            "valid_len": valid_len,
            "sep_idx": sep_idx,
        }
        return (
            sent_enc["input_ids"].squeeze(0),
            sent_enc["attention_mask"].squeeze(0),
            aspect_mask,
            0,
            0,
            "TOKEN_MISMATCH",
            False,
            diag,
        )

    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    has_cls = bool(cls_id is not None and full_ids and full_ids[0] == cls_id)
    has_sep = bool(sep_id is not None and full_ids and full_ids[-1] == sep_id)

    total_len = len(full_ids)
    if total_len <= max_len_sent:
        cropped_ids = list(full_ids)
        new_aspect_start = tok_start
        new_aspect_end = tok_end
    else:
        if has_cls or has_sep:
            content_start = 1 if has_cls else 0
            content_end = total_len - 1 if has_sep else total_len
            max_content_len = max_len_sent - (1 if has_cls else 0) - (1 if has_sep else 0)
            max_content_len = max(1, max_content_len)
            aspect_center = (tok_start + tok_end - 1) // 2
            min_start = content_start
            max_start = max(content_start, content_end - max_content_len)
            crop_content_start = aspect_center - (max_content_len // 2)
            crop_content_start = max(min_start, min(crop_content_start, max_start))
            crop_content_end = crop_content_start + max_content_len
            cropped_ids = []
            if has_cls:
                cropped_ids.append(full_ids[0])
            cropped_ids.extend(full_ids[crop_content_start:crop_content_end])
            if has_sep:
                cropped_ids.append(full_ids[-1])
            # Shift aspect span into the cropped window that excludes specials.
            new_aspect_start = (tok_start - crop_content_start) + (1 if has_cls else 0)
            new_aspect_end = (tok_end - crop_content_start) + (1 if has_cls else 0)
        else:
            aspect_center = (tok_start + tok_end - 1) // 2
            crop_start = aspect_center - (max_len_sent // 2)
            crop_start = max(0, min(crop_start, total_len - max_len_sent))
            crop_end = crop_start + max_len_sent
            cropped_ids = full_ids[crop_start:crop_end]
            new_aspect_start = tok_start - crop_start
            new_aspect_end = tok_end - crop_start

    attention_mask = [1] * len(cropped_ids)
    if len(cropped_ids) < max_len_sent:
        pad_len = max_len_sent - len(cropped_ids)
        cropped_ids.extend([pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)

    aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
    if new_aspect_end > new_aspect_start:
        aspect_mask[new_aspect_start:new_aspect_end] = 1

    sep_idx = -1
    if sep_id is not None:
        try:
            sep_idx = cropped_ids.index(sep_id)
        except ValueError:
            sep_idx = -1
    valid_len = int(sum(attention_mask))
    term_ids = tokenizer(
        term_norm,
        truncation=True,
        padding="max_length",
        max_length=max_len_term,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].squeeze(0).tolist()
    diag = {
        "sentence_norm": sentence_norm,
        "term_norm": term_norm,
        "content_ids": cropped_ids[1:sep_idx] if sep_idx > 1 else [],
        "term_ids": term_ids,
        "sent_tokens": tokenizer.convert_ids_to_tokens(cropped_ids[1:sep_idx])
        if sep_idx > 1
        else [],
        "term_tokens": tokenizer.tokenize((term or "").lower()),
        "token_match_idx": -1,
        "valid_len": valid_len,
        "sep_idx": sep_idx,
    }

    return (
        torch.tensor(cropped_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        aspect_mask,
        int(new_aspect_start),
        int(new_aspect_end),
        "OK",
        True,
        diag,
    )


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Optional[Dict[str, int]] = None,
        mode: Optional[str] = None,
        debug_aspect_span: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Tokenizer must be fast (use_fast=True) for offset mapping.")
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
        self.debug_aspect_span = bool(debug_aspect_span)
        self.include_aspect_span = str(mode or "").strip() == "HAGMoE"

        self.samples = _load_samples(json_path)

        for i, s in enumerate(self.samples[:5]):
            _extract_sample_fields(s, idx=i)

        if label2id is None:
            labels = sorted({_extract_sample_fields(s, idx=i)[2] for i, s in enumerate(self.samples)})
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        sentence, term, label_value = _extract_sample_fields(item, idx=idx)
        label = self.label2id[label_value]

        term_enc = self.tokenizer(
            term,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_term,
            return_tensors="pt",
        )
        if self.include_aspect_span:
            (
                input_ids_sent,
                attention_mask_sent,
                aspect_mask_sent,
                aspect_start,
                aspect_end,
                fail_reason,
                matched,
                diag,
            ) = _compute_aspect_span(
                tokenizer=self.tokenizer,
                term=term,
                sentence=sentence,
                max_len_sent=self.max_len_sent,
                max_len_term=self.max_len_term,
            )

            self.total_samples += 1
            if matched:
                self.matched_samples += 1
                self.matched_mask_sum += float(aspect_mask_sent.sum().item())
            else:
                if fail_reason == "TOKEN_MISMATCH":
                    self.token_mismatch_count += 1
                elif fail_reason == "TRUNCATED":
                    self.truncated_count += 1
                elif fail_reason == "NOT_FOUND_RAW":
                    self.not_found_raw_count += 1

            if self._debug_epoch is not None:
                sample_idx = self._debug_seen
                self._debug_seen += 1

                if (
                    self._debug_epoch == 0
                    and self._debug_split in {"val", "test"}
                    and self._debug_batch_idx == 0
                    and sample_idx < 10
                    and fail_reason != "OK"
                ):
                    sentence_norm = diag.get("sentence_norm", "")
                    term_norm = diag.get("term_norm", "")
                    content_ids = diag.get("content_ids", [])
                    term_ids = diag.get("term_ids", [])
                    valid_len = diag.get("valid_len", 0)
                    sep_idx = diag.get("sep_idx", -1)

                    sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                    term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                    token_match_idx = int(diag.get("token_match_idx", -1))
                    if token_match_idx < 0:
                        token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                    id_match_idx = _find_subsequence(content_ids, term_ids)

                    sent_piece_tokens_view = (
                        sent_piece_tokens
                        if len(sent_piece_tokens) <= 40
                        else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                    )
                    content_ids_view = (
                        content_ids
                        if len(content_ids) <= 80
                        else (content_ids[:60] + ["..."] + content_ids[-20:])
                    )
                    decoded_content = self.tokenizer.decode(content_ids)[:200]
                    decoded_term = self.tokenizer.decode(term_ids)

                    raw_found_idx = -1
                    raw_found = False
                    if term_norm:
                        raw_found_idx = sentence_norm.find(term_norm)
                        raw_found = raw_found_idx >= 0

                    token_found = token_match_idx >= 0
                    reason = ""
                    if raw_found and token_found and id_match_idx < 0:
                        reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                    elif raw_found and not token_found:
                        reason = "TOKENIZATION_SPLIT_DIFF"
                    elif not raw_found:
                        reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                    if valid_len >= self.max_len_sent and raw_found:
                        reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                    def _special_chars_view(s: str) -> str:
                        out = []
                        for ch in s:
                            if not (ch.isalnum() or ch.isspace()):
                                out.append(f"{ch}->U+{ord(ch):04X}")
                        return " ".join(out)

                    sentence_raw_lower = sentence.lower()
                    aspect_raw_lower = term.lower()
                    raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                    raw_substring = (
                        sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                        if raw_idx_in_sentence >= 0
                        else ""
                    )

                    token_span = ""
                    if token_found:
                        token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                    block = [
                        f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                        f"batch={self._debug_batch_idx} sample={sample_idx}",
                        f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                        f"  sentence_raw: {sentence}",
                        f"  aspect_raw: {term}",
                        f"  sentence_norm: {sentence_norm}",
                        f"  aspect_norm: {term_norm}",
                        f"  fail_reason: {fail_reason}",
                        f"  sent_piece_tokens: {sent_piece_tokens_view}",
                        f"  term_piece_tokens: {term_piece_tokens}",
                        f"  sent_piece_ids: {content_ids_view}",
                        f"  term_piece_ids: {term_ids}",
                        f"  decoded_content_snippet: {decoded_content}",
                        f"  decoded_term: {decoded_term}",
                        f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                        f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                        f"  id_found_subsequence: {id_match_idx}",
                        f"  aspect_raw_specials: {_special_chars_view(term)}",
                        f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                        f"  suggested_reason: {reason}",
                    ]
                    print("\n".join(block))

            item_out = {
                "input_ids_sent": input_ids_sent,
                "attention_mask_sent": attention_mask_sent,
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
                "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
                "aspect_mask_sent": aspect_mask_sent,
                "label": torch.tensor(label, dtype=torch.long),
            }

            if self.debug_aspect_span:
                sent_ids = input_ids_sent.tolist()
                sep_id = getattr(self.tokenizer, "sep_token_id", None)
                sep_idx = -1
                if sep_id is not None:
                    try:
                        sep_idx = sent_ids.index(sep_id)
                    except ValueError:
                        sep_idx = -1
                valid_len = int(attention_mask_sent.sum().item())
                sentence_raw = item.get("sentence_raw", sentence)
                aspect_raw = item.get("aspect_raw", term)

                item_out.update(
                    {
                        "sentence_raw": sentence_raw,
                        "aspect_raw": aspect_raw,
                        "valid_len": torch.tensor(valid_len, dtype=torch.long),
                        "sep_idx": torch.tensor(sep_idx, dtype=torch.long),
                        "max_len_sent": torch.tensor(self.max_len_sent, dtype=torch.long),
                    }
                )
        else:
            sent_enc = self.tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_sent,
                return_tensors="pt",
            )
            item_out = {
                "input_ids_sent": sent_enc["input_ids"].squeeze(0),
                "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

        return item_out


def _majority_with_tiebreak(
    pols: Sequence[str],
    prefer_order_no_neu: Tuple[str, ...] = ("positive", "negative", "neutral"),
) -> str:
    c = Counter(pols)
    if not c:
        return "neutral"

    max_cnt = max(c.values())
    tied = [p for p, cnt in c.items() if cnt == max_cnt]

    if len(tied) == 1:
        return tied[0]

    if "neutral" in tied:
        return "neutral"

    for p in prefer_order_no_neu:
        if p in tied:
            return p

    return sorted(tied)[0]


class _SubsetAspectSentimentDataset(Dataset):
    def __init__(
        self,
        base_samples: List[dict],
        indices: List[int],
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Dict[str, int],
        mode: Optional[str] = None,
        debug_aspect_span: bool = False,
    ) -> None:
        self._base_samples = base_samples
        self._indices = indices
        self.tokenizer = tokenizer
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Tokenizer must be fast (use_fast=True) for offset mapping.")
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
        self.label2id = label2id
        self.debug_aspect_span = bool(debug_aspect_span)
        self.include_aspect_span = str(mode or "").strip() == "HAGMoE"

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def __len__(self) -> int:
        return len(self._indices)

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._indices[idx]
        item = self._base_samples[real_idx]

        sentence, term, label_value = _extract_sample_fields(item, idx=real_idx)
        label = self.label2id[label_value]

        term_enc = self.tokenizer(
            term,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_term,
            return_tensors="pt",
        )
        if self.include_aspect_span:
            (
                input_ids_sent,
                attention_mask_sent,
                aspect_mask_sent,
                aspect_start,
                aspect_end,
                fail_reason,
                matched,
                diag,
            ) = _compute_aspect_span(
                tokenizer=self.tokenizer,
                term=term,
                sentence=sentence,
                max_len_sent=self.max_len_sent,
                max_len_term=self.max_len_term,
            )

            self.total_samples += 1
            if matched:
                self.matched_samples += 1
                self.matched_mask_sum += float(aspect_mask_sent.sum().item())
            else:
                if fail_reason == "TOKEN_MISMATCH":
                    self.token_mismatch_count += 1
                elif fail_reason == "TRUNCATED":
                    self.truncated_count += 1
                elif fail_reason == "NOT_FOUND_RAW":
                    self.not_found_raw_count += 1

            if self._debug_epoch is not None:
                sample_idx = self._debug_seen
                self._debug_seen += 1

                if (
                    self._debug_epoch == 0
                    and self._debug_split in {"val", "test"}
                    and self._debug_batch_idx == 0
                    and sample_idx < 10
                    and fail_reason != "OK"
                ):
                    sentence_norm = diag.get("sentence_norm", "")
                    term_norm = diag.get("term_norm", "")
                    content_ids = diag.get("content_ids", [])
                    term_ids = diag.get("term_ids", [])
                    valid_len = diag.get("valid_len", 0)
                    sep_idx = diag.get("sep_idx", -1)

                    sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                    term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                    token_match_idx = int(diag.get("token_match_idx", -1))
                    if token_match_idx < 0:
                        token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                    id_match_idx = _find_subsequence(content_ids, term_ids)

                    sent_piece_tokens_view = (
                        sent_piece_tokens
                        if len(sent_piece_tokens) <= 40
                        else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                    )
                    content_ids_view = (
                        content_ids
                        if len(content_ids) <= 80
                        else (content_ids[:60] + ["..."] + content_ids[-20:])
                    )
                    decoded_content = self.tokenizer.decode(content_ids)[:200]
                    decoded_term = self.tokenizer.decode(term_ids)

                    raw_found_idx = -1
                    raw_found = False
                    if term_norm:
                        raw_found_idx = sentence_norm.find(term_norm)
                        raw_found = raw_found_idx >= 0

                    token_found = token_match_idx >= 0
                    reason = ""
                    if raw_found and token_found and id_match_idx < 0:
                        reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                    elif raw_found and not token_found:
                        reason = "TOKENIZATION_SPLIT_DIFF"
                    elif not raw_found:
                        reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                    if valid_len >= self.max_len_sent and raw_found:
                        reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                    def _special_chars_view(s: str) -> str:
                        out = []
                        for ch in s:
                            if not (ch.isalnum() or ch.isspace()):
                                out.append(f"{ch}->U+{ord(ch):04X}")
                        return " ".join(out)

                    sentence_raw_lower = sentence.lower()
                    aspect_raw_lower = term.lower()
                    raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                    raw_substring = (
                        sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                        if raw_idx_in_sentence >= 0
                        else ""
                    )

                    token_span = ""
                    if token_found:
                        token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                    block = [
                        f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                        f"batch={self._debug_batch_idx} sample={sample_idx}",
                        f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                        f"  sentence_raw: {sentence}",
                        f"  aspect_raw: {term}",
                        f"  sentence_norm: {sentence_norm}",
                        f"  aspect_norm: {term_norm}",
                        f"  fail_reason: {fail_reason}",
                        f"  sent_piece_tokens: {sent_piece_tokens_view}",
                        f"  term_piece_tokens: {term_piece_tokens}",
                        f"  sent_piece_ids: {content_ids_view}",
                        f"  term_piece_ids: {term_ids}",
                        f"  decoded_content_snippet: {decoded_content}",
                        f"  decoded_term: {decoded_term}",
                        f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                        f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                        f"  id_found_subsequence: {id_match_idx}",
                        f"  aspect_raw_specials: {_special_chars_view(term)}",
                        f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                        f"  suggested_reason: {reason}",
                    ]
                    print("\n".join(block))

            item_out = {
                "input_ids_sent": input_ids_sent,
                "attention_mask_sent": attention_mask_sent,
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
                "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
                "aspect_mask_sent": aspect_mask_sent,
                "label": torch.tensor(label, dtype=torch.long),
            }

            if self.debug_aspect_span:
                sent_ids = input_ids_sent.tolist()
                sep_id = getattr(self.tokenizer, "sep_token_id", None)
                sep_idx = -1
                if sep_id is not None:
                    try:
                        sep_idx = sent_ids.index(sep_id)
                    except ValueError:
                        sep_idx = -1
                valid_len = int(attention_mask_sent.sum().item())
                sentence_raw = item.get("sentence_raw", sentence)
                aspect_raw = item.get("aspect_raw", term)

                item_out.update(
                    {
                        "sentence_raw": sentence_raw,
                        "aspect_raw": aspect_raw,
                        "valid_len": torch.tensor(valid_len, dtype=torch.long),
                        "sep_idx": torch.tensor(sep_idx, dtype=torch.long),
                        "max_len_sent": torch.tensor(self.max_len_sent, dtype=torch.long),
                    }
                )
        else:
            sent_enc = self.tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_sent,
                return_tensors="pt",
            )
            item_out = {
                "input_ids_sent": sent_enc["input_ids"].squeeze(0),
                "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

        return item_out

    @property
    def base_indices(self) -> List[int]:
        return self._indices


class AspectSentimentDatasetKFold(Dataset):
    """
    Full-train dataset with sentence-level K-fold split (phân rã KFoldConfig).

    Parameters:
        k_folds: số fold, ví dụ 5
        seed: seed dùng cho shuffle sentence
        shuffle: có shuffle sentence indices trước khi chia fold hay không

    Usage:
        base = AspectSentimentDatasetKFold(
            json_path="train.json",
            tokenizer=tok,
            max_len_sent=128,
            max_len_term=16,
            k_folds=config.k_folds,
            seed=config.seed,
            shuffle=True,
        )
        train_ds, val_ds = base.get_fold(0)
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        k_folds: int,
        seed: int,
        shuffle: bool = True,
        label2id: Optional[Dict[str, int]] = None,
        mode: Optional[str] = None,
        debug_aspect_span: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Tokenizer must be fast (use_fast=True) for offset mapping.")
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
        self.debug_aspect_span = bool(debug_aspect_span)
        self.include_aspect_span = str(mode or "").strip() == "HAGMoE"

        self.k_folds = int(k_folds)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        self.samples: List[dict] = _load_samples(json_path)

        if label2id is None:
            labels = sorted({_extract_sample_fields(s, idx=i)[2] for i, s in enumerate(self.samples)})
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

        (
            self._sent_list,
            self._sent_to_row_indices,
            self._sent_strata,
        ) = self._build_sentence_groups_and_strata(self.samples)

        self._folds = self._build_folds(
            n_sents=len(self._sent_list),
            strata=self._sent_strata,
            k_folds=self.k_folds,
            seed=self.seed,
            shuffle=self.shuffle,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        sentence, term, label_value = _extract_sample_fields(item, idx=idx)
        label = self.label2id[label_value]

        term_enc = self.tokenizer(
            term,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_term,
            return_tensors="pt",
        )

        if self.include_aspect_span:
            (
                input_ids_sent,
                attention_mask_sent,
                aspect_mask_sent,
                aspect_start,
                aspect_end,
                fail_reason,
                matched,
                diag,
            ) = _compute_aspect_span(
                tokenizer=self.tokenizer,
                term=term,
                sentence=sentence,
                max_len_sent=self.max_len_sent,
                max_len_term=self.max_len_term,
            )

            self.total_samples += 1
            if matched:
                self.matched_samples += 1
                self.matched_mask_sum += float(aspect_mask_sent.sum().item())
            else:
                if fail_reason == "TOKEN_MISMATCH":
                    self.token_mismatch_count += 1
                elif fail_reason == "TRUNCATED":
                    self.truncated_count += 1
                elif fail_reason == "NOT_FOUND_RAW":
                    self.not_found_raw_count += 1

            if self._debug_epoch is not None:
                sample_idx = self._debug_seen
                self._debug_seen += 1

                if (
                    self._debug_epoch == 0
                    and self._debug_split in {"val", "test"}
                    and self._debug_batch_idx == 0
                    and sample_idx < 10
                    and fail_reason != "OK"
                ):
                    sentence_norm = diag.get("sentence_norm", "")
                    term_norm = diag.get("term_norm", "")
                    content_ids = diag.get("content_ids", [])
                    term_ids = diag.get("term_ids", [])
                    valid_len = diag.get("valid_len", 0)
                    sep_idx = diag.get("sep_idx", -1)

                    sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                    term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                    token_match_idx = int(diag.get("token_match_idx", -1))
                    if token_match_idx < 0:
                        token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                    id_match_idx = _find_subsequence(content_ids, term_ids)

                    sent_piece_tokens_view = (
                        sent_piece_tokens
                        if len(sent_piece_tokens) <= 40
                        else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                    )
                    content_ids_view = (
                        content_ids
                        if len(content_ids) <= 80
                        else (content_ids[:60] + ["..."] + content_ids[-20:])
                    )
                    decoded_content = self.tokenizer.decode(content_ids)[:200]
                    decoded_term = self.tokenizer.decode(term_ids)

                    raw_found_idx = -1
                    raw_found = False
                    if term_norm:
                        raw_found_idx = sentence_norm.find(term_norm)
                        raw_found = raw_found_idx >= 0

                    token_found = token_match_idx >= 0
                    reason = ""
                    if raw_found and token_found and id_match_idx < 0:
                        reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                    elif raw_found and not token_found:
                        reason = "TOKENIZATION_SPLIT_DIFF"
                    elif not raw_found:
                        reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                    if valid_len >= self.max_len_sent and raw_found:
                        reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                    def _special_chars_view(s: str) -> str:
                        out = []
                        for ch in s:
                            if not (ch.isalnum() or ch.isspace()):
                                out.append(f"{ch}->U+{ord(ch):04X}")
                        return " ".join(out)

                    sentence_raw_lower = sentence.lower()
                    aspect_raw_lower = term.lower()
                    raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                    raw_substring = (
                        sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                        if raw_idx_in_sentence >= 0
                        else ""
                    )

                    token_span = ""
                    if token_found:
                        token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                    block = [
                        f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                        f"batch={self._debug_batch_idx} sample={sample_idx}",
                        f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                        f"  sentence_raw: {sentence}",
                        f"  aspect_raw: {term}",
                        f"  sentence_norm: {sentence_norm}",
                        f"  aspect_norm: {term_norm}",
                        f"  fail_reason: {fail_reason}",
                        f"  sent_piece_tokens: {sent_piece_tokens_view}",
                        f"  term_piece_tokens: {term_piece_tokens}",
                        f"  sent_piece_ids: {content_ids_view}",
                        f"  term_piece_ids: {term_ids}",
                        f"  decoded_content_snippet: {decoded_content}",
                        f"  decoded_term: {decoded_term}",
                        f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                        f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                        f"  id_found_subsequence: {id_match_idx}",
                        f"  aspect_raw_specials: {_special_chars_view(term)}",
                        f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                        f"  suggested_reason: {reason}",
                    ]
                    print("\n".join(block))

            item_out = {
                "input_ids_sent": input_ids_sent,
                "attention_mask_sent": attention_mask_sent,
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
                "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
                "aspect_mask_sent": aspect_mask_sent,
                "label": torch.tensor(label, dtype=torch.long),
            }

            if self.debug_aspect_span:
                sent_ids = input_ids_sent.tolist()
                sep_id = getattr(self.tokenizer, "sep_token_id", None)
                sep_idx = -1
                if sep_id is not None:
                    try:
                        sep_idx = sent_ids.index(sep_id)
                    except ValueError:
                        sep_idx = -1
                valid_len = int(attention_mask_sent.sum().item())
                sentence_raw = item.get("sentence_raw", sentence)
                aspect_raw = item.get("aspect_raw", term)

                item_out.update(
                    {
                        "sentence_raw": sentence_raw,
                        "aspect_raw": aspect_raw,
                        "valid_len": torch.tensor(valid_len, dtype=torch.long),
                        "sep_idx": torch.tensor(sep_idx, dtype=torch.long),
                        "max_len_sent": torch.tensor(self.max_len_sent, dtype=torch.long),
                    }
                )
        else:
            sent_enc = self.tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_sent,
                return_tensors="pt",
            )
            item_out = {
                "input_ids_sent": sent_enc["input_ids"].squeeze(0),
                "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
                "input_ids_term": term_enc["input_ids"].squeeze(0),
                "attention_mask_term": term_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

        return item_out

    def num_folds(self) -> int:
        return len(self._folds)

    def get_fold(self, fold_idx: int) -> Tuple[Dataset, Dataset]:
        if fold_idx < 0 or fold_idx >= len(self._folds):
            raise IndexError(f"fold_idx out of range: {fold_idx}")

        val_sent_indices = set(self._folds[fold_idx])
        train_sent_indices = [i for i in range(len(self._sent_list)) if i not in val_sent_indices]

        train_row_indices = self._sent_indices_to_row_indices(train_sent_indices)
        val_row_indices = self._sent_indices_to_row_indices(sorted(val_sent_indices))

        train_ds = _SubsetAspectSentimentDataset(
            base_samples=self.samples,
            indices=train_row_indices,
            tokenizer=self.tokenizer,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
            label2id=self.label2id,
            mode="HAGMoE" if self.include_aspect_span else None,
            debug_aspect_span=self.debug_aspect_span,
        )
        val_ds = _SubsetAspectSentimentDataset(
            base_samples=self.samples,
            indices=val_row_indices,
            tokenizer=self.tokenizer,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
            label2id=self.label2id,
            mode="HAGMoE" if self.include_aspect_span else None,
            debug_aspect_span=self.debug_aspect_span,
        )
        return train_ds, val_ds

    def _sent_indices_to_row_indices(self, sent_indices: List[int]) -> List[int]:
        out: List[int] = []
        for si in sent_indices:
            sent = self._sent_list[si]
            out.extend(self._sent_to_row_indices[sent])
        return out

    @staticmethod
    def _build_sentence_groups_and_strata(
        samples: List[dict],
    ) -> Tuple[List[str], Dict[str, List[int]], List[str]]:
        sent_to_rows: Dict[str, List[int]] = defaultdict(list)
        sent_to_pols: Dict[str, List[str]] = defaultdict(list)

        for i, s in enumerate(samples):
            sent, _, pol = _extract_sample_fields(s, idx=i)
            sent_to_rows[sent].append(i)
            sent_to_pols[sent].append(str(pol))

        sent_list = list(sent_to_rows.keys())

        sent_strata: List[str] = []
        for sent in sent_list:
            pols = sent_to_pols[sent]
            major = _majority_with_tiebreak(pols)
            has_neu = any(p == "neutral" for p in pols)
            strata_key = f"{major}|neu{1 if has_neu else 0}"
            sent_strata.append(strata_key)

        return sent_list, dict(sent_to_rows), sent_strata

    @staticmethod
    def _build_folds(
        n_sents: int,
        strata: List[str],
        k_folds: int,
        seed: int,
        shuffle: bool,
    ) -> List[List[int]]:
        if k_folds <= 1:
            raise ValueError("k_folds must be >= 2 for K-fold")

        if n_sents < k_folds:
            raise ValueError(f"Not enough sentences for k-fold: n_sents={n_sents} < k_folds={k_folds}")

        counts = Counter(strata)
        can_stratify = all(v >= k_folds for v in counts.values())

        rng = random.Random(seed)
        indices = list(range(n_sents))
        if shuffle:
            rng.shuffle(indices)

        if not can_stratify:
            folds: List[List[int]] = [[] for _ in range(k_folds)]
            for j, si in enumerate(indices):
                folds[j % k_folds].append(si)
            return folds

        strata_to_indices: Dict[str, List[int]] = defaultdict(list)
        for si in indices:
            strata_to_indices[strata[si]].append(si)

        folds = [[] for _ in range(k_folds)]
        for _, group in strata_to_indices.items():
            for j, si in enumerate(group):
                folds[j % k_folds].append(si)

        if shuffle:
            for f in folds:
                rng.shuffle(f)

        return folds
