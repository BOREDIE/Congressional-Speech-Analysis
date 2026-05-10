import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PANEL_DIR = _REPO_ROOT / "01_data" / "02_Panel"
_DEFAULT_SELECTED_WORDS = _REPO_ROOT / "01_data" / "03_TargetWords" / "selected_words_33.csv"

MODEL_NAME = "ddore14/RooseBERT-cont-cased"
TEXT_COLUMN = "speech_text"
ID_COLUMN = "speech_id"
WORD_RE = re.compile(r"\b\w+\b")
PARTICIPANT_COLUMNS = [
    "bioguide_id",
    "party",
    "state",
    "district",
    "chamber",
    "matched_pair_id",
    "cohort",
    "match_weight",
]


def normalize_group_value(value):
    """Coerce a possibly-NA value to a non-null string."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def merge_metadata_values(values):
    """Join a collection of string values into a sorted pipe-delimited string."""
    cleaned = sorted({value for value in values if value != ""})
    return "|".join(cleaned)


def word_spans(text):
    """Return regex word spans as (word_index, word, start_char, end_char)."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return [(i, m.group(), m.start(), m.end()) for i, m in enumerate(WORD_RE.finditer(text))]


def find_word_index(spans, token_start, token_end, cursor):
    """Find the word span that overlaps a token offset."""
    while cursor < len(spans) and spans[cursor][3] <= token_start:
        cursor += 1
    if cursor < len(spans):
        _, _, word_start, word_end = spans[cursor]
        if token_start < word_end and token_end > word_start:
            return cursor, cursor
    return None, cursor


def flush_shard(out_dir, shard_id, metadata_rows, embedding_rows, parquet=True):
    """Write a shard of embeddings and their metadata to disk."""
    if not metadata_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embedding_rows, dtype=np.float16)  # store as float16 to halve disk usage

    emb_path = out_dir / f"embeddings_{shard_id:05d}.npy"
    meta_path = out_dir / f"metadata_{shard_id:05d}.parquet"
    fallback_meta_path = out_dir / f"metadata_{shard_id:05d}.csv"

    np.save(emb_path, embeddings)
    metadata = pd.DataFrame(metadata_rows)
    if parquet:
        try:
            metadata.to_parquet(meta_path, index=False)
        except Exception:
            metadata.to_csv(fallback_meta_path, index=False)
    else:
        metadata.to_csv(fallback_meta_path, index=False)



def write_vocab_embeddings(out_dir, vocab_sums, vocab_counts, parquet=True):
    """Average token-level sums into vocabulary embeddings and write to disk."""
    if not vocab_sums:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    words = sorted(vocab_sums)
    embeddings = np.asarray(
        [vocab_sums[word] / vocab_counts[word] for word in words],
        dtype=np.float16,
    )
    metadata = pd.DataFrame(
        {
            "word": words,
            "occurrence_count": [int(vocab_counts[word]) for word in words],
        }
    )

    np.save(out_dir / "vocab_embeddings.npy", embeddings)
    if parquet:
        try:
            metadata.to_parquet(out_dir / "vocab_metadata.parquet", index=False)
        except Exception:
            metadata.to_csv(out_dir / "vocab_metadata.csv", index=False)
    else:
        metadata.to_csv(out_dir / "vocab_metadata.csv", index=False)



def load_target_words(selected_words_path):
    """Load the selected target-word CSV and build a lower-cased lookup dict."""
    selected_words = pd.read_csv(selected_words_path)
    if "word" not in selected_words.columns:
        raise ValueError("selected words CSV must contain a 'word' column.")

    has_frame = "frame" in selected_words.columns
    lookup = {}
    for _, row in selected_words.iterrows():
        word = str(row["word"]).strip()
        if not word:
            continue
        lookup[word.lower()] = {
            "target_word": word,
            "frame": row["frame"] if has_frame else None,
        }
    return lookup


def write_target_group_embeddings(
    out_dir,
    group_sums,
    group_counts,
    group_metadata_values=None,
    parquet=True,
):
    """Write averaged target-word embeddings keyed by (frame, word, group, speaker, period)."""
    if not group_sums:
        print("WARNING: no selected target word embeddings found")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(group_sums)
    embeddings = np.asarray(
        [group_sums[key] / group_counts[key] for key in keys],
        dtype=np.float16,
    )
    metadata_rows = []
    for key in keys:
        row = {
                "frame": key[0],
                "target_word": key[1],
                "group": key[2],
                "speaker": key[3],
                "period": key[4],
                "occurrence_count": int(group_counts[key]),
            }
        if group_metadata_values is not None:
            for column in PARTICIPANT_COLUMNS:
                row[column] = merge_metadata_values(
                    group_metadata_values.get(key, {}).get(column, set())
                )
        metadata_rows.append(row)
    metadata = pd.DataFrame(metadata_rows)

    np.save(out_dir / "target_word_speaker_period_embeddings.npy", embeddings)
    if parquet:
        try:
            metadata.to_parquet(
                out_dir / "target_word_speaker_period_metadata.parquet",
                index=False,
            )
        except Exception:
            metadata.to_csv(
                out_dir / "target_word_speaker_period_metadata.csv",
                index=False,
            )
    else:
        metadata.to_csv(
            out_dir / "target_word_speaker_period_metadata.csv",
            index=False,
        )

    print(f"Saved: target word speaker-period embeddings ({len(keys):,} groups) → {out_dir}")


def update_vocab_aggregation(vocab_sums, vocab_counts, metadata_rows, embedding_rows, normalization):
    """Accumulate running sum/count for each word form to support average-pooling later."""
    for metadata, embedding in zip(metadata_rows, embedding_rows):
        word = metadata["word"]
        if normalization == "lower":
            word = word.lower()

        if word not in vocab_sums:
            vocab_sums[word] = np.zeros_like(embedding, dtype=np.float32)
            vocab_counts[word] = 0

        vocab_sums[word] += embedding.astype(np.float32, copy=False)
        vocab_counts[word] += 1


def update_target_group_aggregation(
    group_sums,
    group_counts,
    group_metadata_values,
    metadata_rows,
    embedding_rows,
):
    """Accumulate running sum/count for each (frame, word, group, speaker, period) key."""
    for metadata, embedding in zip(metadata_rows, embedding_rows):
        key = (
            metadata["frame"],
            metadata["target_word"],
            metadata["group"],
            metadata["speaker"],
            metadata["period"],
        )
        if key not in group_sums:
            group_sums[key] = np.zeros_like(embedding, dtype=np.float32)
            group_counts[key] = 0
            group_metadata_values[key] = {column: set() for column in PARTICIPANT_COLUMNS}

        group_sums[key] += embedding.astype(np.float32, copy=False)
        group_counts[key] += 1
        for column in PARTICIPANT_COLUMNS:
            value = normalize_group_value(metadata.get(column, ""))
            if value:
                group_metadata_values[key][column].add(value)


def embed_speech_batch(
    rows,
    tokenizer,
    model,
    device,
    max_length,
    stride,
    layer,
    layer_pooling,
    target_lookup=None,
    speaker_column=None,
    period_column=None,
    period_fallback_column=None,
):
    """Tokenize a batch of speeches, run the model, and return per-word embeddings."""
    texts = ["" if pd.isna(row[TEXT_COLUMN]) else str(row[TEXT_COLUMN]) for _, row in rows]
    speech_ids = [row.get(ID_COLUMN, row_index) for row_index, row in rows]
    groups = [
        normalize_group_value(row.get("_corpus_group", "")) if "_corpus_group" in row else ""
        for _, row in rows
    ]
    speakers = [
        normalize_group_value(row.get(speaker_column, None)) if speaker_column else ""
        for _, row in rows
    ]
    periods = []
    for _, row in rows:
        value = row.get(period_column, None) if period_column else None
        if (value is None or pd.isna(value)) and period_fallback_column:
            value = row.get(period_fallback_column, None)
        periods.append(normalize_group_value(value))
    participant_metadata = [
        {
            column: normalize_group_value(row.get(column, ""))
            for column in PARTICIPANT_COLUMNS
        }
        for _, row in rows
    ]
    all_spans = [word_spans(text) for text in texts]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
    offset_mapping = encoded.pop("offset_mapping").cpu().numpy()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)
    if layer_pooling == "last4":
        last_four = output.hidden_states[-4:]
        hidden = torch.stack(last_four, dim=0).mean(dim=0)  # mean of last 4 layers
    elif layer_pooling == "last":
        hidden = output.last_hidden_state
    else:
        hidden = output.hidden_states[layer]   # single specified layer
    hidden = hidden.detach().cpu().numpy()

    sums = [np.zeros((len(spans), hidden.shape[-1]), dtype=np.float32) for spans in all_spans]
    counts = [np.zeros(len(spans), dtype=np.int32) for spans in all_spans]
    seen_offsets = [set() for _ in all_spans]

    for window_index, sample_index in enumerate(sample_mapping):
        spans = all_spans[sample_index]
        cursor = 0
        for token_index, (token_start, token_end) in enumerate(offset_mapping[window_index]):
            if token_start == token_end:
                continue

            offset_key = (int(token_start), int(token_end))
            if offset_key in seen_offsets[sample_index]:
                continue
            seen_offsets[sample_index].add(offset_key)

            word_index, cursor = find_word_index(spans, token_start, token_end, cursor)
            if word_index is None:
                continue

            sums[sample_index][word_index] += hidden[window_index, token_index]
            counts[sample_index][word_index] += 1

    metadata_rows = []
    embedding_rows = []
    for sample_index, spans in enumerate(all_spans):
        speech_id = speech_ids[sample_index]
        valid = counts[sample_index] > 0
        if not valid.any():
            continue

        word_embeddings = sums[sample_index][valid] / counts[sample_index][valid, None]
        valid_indices = np.flatnonzero(valid)
        for embedding, span_index in zip(word_embeddings, valid_indices):
            word_index, word, start_char, end_char = spans[span_index]
            target_info = None
            if target_lookup is not None:
                target_info = target_lookup.get(word.lower())
                if target_info is None:
                    continue

            metadata_rows.append(
                {
                    "speech_id": speech_id,
                    "word_index": int(word_index),
                    "word": word,
                    "start_char": int(start_char),
                    "end_char": int(end_char),
                    "frame": target_info["frame"] if target_info else None,
                    "target_word": target_info["target_word"] if target_info else None,
                    "group": groups[sample_index],
                    "speaker": speakers[sample_index],
                    "period": periods[sample_index],
                    **participant_metadata[sample_index],
                }
            )
            embedding_rows.append(embedding)

    return metadata_rows, embedding_rows


def iter_csv_batches(csv_path, chunksize, batch_size, limit_rows):
    """Yield individual rows from a single CSV, respecting an optional row limit."""
    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if TEXT_COLUMN not in chunk.columns:
            raise ValueError(f"CSV must contain a '{TEXT_COLUMN}' column.")

        for _, row in chunk.iterrows():
            if limit_rows is not None and total_rows >= limit_rows:
                return
            total_rows += 1
            yield row


def parse_grouped_csv(value):
    """Parse a 'label=path' string into (label, Path); falls back to using the filename stem."""
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    group, path = value.split("=", 1)
    return group.strip(), Path(path.strip())


def iter_grouped_csv_batches(grouped_csvs, chunksize, limit_rows):
    """Yield rows from multiple labelled CSVs, tagging each row with its corpus group."""
    total_rows = 0
    for group, csv_path in grouped_csvs:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if TEXT_COLUMN not in chunk.columns:
                raise ValueError(f"{csv_path} must contain a '{TEXT_COLUMN}' column.")

            chunk = chunk.copy()
            chunk["_corpus_group"] = group
            for _, row in chunk.iterrows():
                if limit_rows is not None and total_rows >= limit_rows:
                    return
                total_rows += 1
                yield row


def main():
    """Parse CLI args, load model, iterate speeches, and write word-level embeddings."""
    parser = argparse.ArgumentParser(
        description="Compute word-level RooseBERT embeddings for congress speech_text."
    )
    parser.add_argument("--csv", default="congress_speech_corpus_updated0430.csv")
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=None,
        help=(
            "One or more small corpora for target-speaker-period mode. "
            "Use label=path, for example treatment=01_data/02_Panel/treatment_speech_corpus.csv control=01_data/02_Panel/control_corpus.csv."
        ),
    )
    parser.add_argument("--out-dir", default="roosebert_target_word_speaker_period_embeddings")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=200_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--layer-pooling",
        choices=["last4", "last", "single"],
        default="last4",
        help="Which hidden states to use. Default averages the last 4 layers.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Used only with --layer-pooling single. 0 is input embeddings; 12 is last layer for BERT-base.",
    )
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--csv-metadata", action="store_true")
    parser.add_argument(
        "--output-mode",
        choices=["target-speaker-period", "occurrences", "vocab"],
        default="target-speaker-period",
        help=(
            "target-speaker-period averages selected target-word occurrences by speaker and period; "
            "occurrences saves one embedding per word occurrence; vocab averages all occurrences with the same word form."
        ),
    )
    parser.add_argument(
        "--word-normalization",
        choices=["surface", "lower"],
        default="surface",
        help="Used only with --output-mode vocab. surface keeps Bank and bank separate; lower merges them.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use cached Hugging Face files only after the model has already been downloaded.",
    )
    parser.add_argument(
        "--selected-words",
        default=_DEFAULT_SELECTED_WORDS,
        help="CSV containing selected target words in a 'word' column, optionally with a 'frame' column.",
    )
    parser.add_argument(
        "--speaker-column",
        default="speaker_name",
        help="Column used for speaker grouping in target-speaker-period mode.",
    )
    parser.add_argument(
        "--period-column",
        default="period",
        help="Column used for period grouping in target-speaker-period mode.",
    )
    parser.add_argument(
        "--period-fallback-column",
        default="congress",
        help="Fallback period column when --period-column is missing or empty.",
    )
    args = parser.parse_args()
    print("Loading model and corpora …")

    csv_path = Path(args.csv)
    grouped_csvs = [parse_grouped_csv(value) for value in args.input_csvs or []]
    if args.output_mode == "target-speaker-period" and not grouped_csvs:
        default_grouped_csvs = [
            ("treatment", _DEFAULT_PANEL_DIR / "treatment_speech_corpus.csv"),
            ("control", _DEFAULT_PANEL_DIR / "control_corpus.csv"),
        ]
        if all(path.exists() for _, path in default_grouped_csvs):
            grouped_csvs = default_grouped_csvs
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_lookup = None
    if args.output_mode == "target-speaker-period":
        target_lookup = load_target_words(args.selected_words)
        csvs_to_check = grouped_csvs or [("", csv_path)]
        for group, path in csvs_to_check:
            csv_columns = pd.read_csv(path, nrows=0).columns
            missing_columns = [
                column
                for column in [TEXT_COLUMN, args.speaker_column]
                if column not in csv_columns
            ]
            has_period = args.period_column in csv_columns
            has_fallback = args.period_fallback_column in csv_columns
            if not has_period and not has_fallback:
                missing_columns.append(
                    f"{args.period_column} or {args.period_fallback_column}"
                )
            if missing_columns:
                label = f"{group} " if group else ""
                raise ValueError(f"{label}CSV is missing required column(s): {missing_columns}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        add_pooling_layer=False,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    metadata_buffer = []
    embedding_buffer = []
    vocab_sums = {}
    vocab_counts = {}
    group_sums = {}
    group_counts = {}
    group_metadata_values = {}
    batch = []
    shard_id = 0
    processed = 0

    row_iter = (
        iter_grouped_csv_batches(grouped_csvs, args.chunksize, args.limit_rows)
        if grouped_csvs
        else iter_csv_batches(csv_path, args.chunksize, args.batch_size, args.limit_rows)
    )
    for row_index, row in enumerate(row_iter):
        batch.append((row_index, row))
        if len(batch) < args.batch_size:
            continue

        metadata_rows, embedding_rows = embed_speech_batch(
            batch,
            tokenizer,
            model,
            device,
            args.max_length,
            args.stride,
            args.layer,
            args.layer_pooling,
            target_lookup=target_lookup,
            speaker_column=args.speaker_column,
            period_column=args.period_column,
            period_fallback_column=args.period_fallback_column,
        )

        if args.output_mode == "vocab":
            update_vocab_aggregation(
                vocab_sums,
                vocab_counts,
                metadata_rows,
                embedding_rows,
                args.word_normalization,
            )
        elif args.output_mode == "target-speaker-period":
            update_target_group_aggregation(
                group_sums,
                group_counts,
                group_metadata_values,
                metadata_rows,
                embedding_rows,
            )
        else:
            metadata_buffer.extend(metadata_rows)
            embedding_buffer.extend(embedding_rows)
        processed += len(batch)
        batch = []

        while args.output_mode == "occurrences" and len(metadata_buffer) >= args.shard_size:
            flush_shard(
                out_dir,
                shard_id,
                metadata_buffer[: args.shard_size],
                embedding_buffer[: args.shard_size],
                parquet=not args.csv_metadata,
            )
            del metadata_buffer[: args.shard_size]
            del embedding_buffer[: args.shard_size]
            shard_id += 1

    if batch:
        metadata_rows, embedding_rows = embed_speech_batch(
            batch,
            tokenizer,
            model,
            device,
            args.max_length,
            args.stride,
            args.layer,
            args.layer_pooling,
            target_lookup=target_lookup,
            speaker_column=args.speaker_column,
            period_column=args.period_column,
            period_fallback_column=args.period_fallback_column,
        )
        if args.output_mode == "vocab":
            update_vocab_aggregation(
                vocab_sums,
                vocab_counts,
                metadata_rows,
                embedding_rows,
                args.word_normalization,
            )
        elif args.output_mode == "target-speaker-period":
            update_target_group_aggregation(
                group_sums,
                group_counts,
                group_metadata_values,
                metadata_rows,
                embedding_rows,
            )
        else:
            metadata_buffer.extend(metadata_rows)
            embedding_buffer.extend(embedding_rows)
        processed += len(batch)

    if args.output_mode == "vocab":
        write_vocab_embeddings(
            out_dir,
            vocab_sums,
            vocab_counts,
            parquet=not args.csv_metadata,
        )
    elif args.output_mode == "target-speaker-period":
        write_target_group_embeddings(
            out_dir,
            group_sums,
            group_counts,
            group_metadata_values=group_metadata_values,
            parquet=not args.csv_metadata,
        )
    else:
        flush_shard(
            out_dir,
            shard_id,
            metadata_buffer,
            embedding_buffer,
            parquet=not args.csv_metadata,
        )
    print(f"done. processed {processed:,} speeches. output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
