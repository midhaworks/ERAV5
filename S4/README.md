# S4 — Data cleaning and deduplication

Interactive assignment report for an eight-stage audit of
`SupraLabs/reasoning-summaries-61k`.

## Dataset-size evidence

The assignment's `10–100M` requirement is interpreted as a decimal
10–100 MB source-file band. The selected dataset satisfies it in two
independent views:

- Hugging Face lists `dataset.jsonl` as 79.4 MB.
- The downloaded audit input measures exactly 7,93,68,728 UTF-8 bytes
  (79.368728 MB), with its SHA-256 recorded in the cleanup report.

Therefore:

`1,00,00,000 ≤ 7,93,68,728 ≤ 10,00,00,000 bytes`

The file evidence is available at:
https://huggingface.co/datasets/SupraLabs/reasoning-summaries-61k/tree/main

## Measured result

- Rows: 61,000 → 57,528; 3,472 quarantined (5.69%)
- JSONL: 7,93,68,728 → 7,54,45,546 bytes (4.94% reduction)
- Training-field characters: 7,56,58,341 → 7,21,54,409 (4.63% reduction)
- Approximate tokens at four characters per token:
  1,89,14,585 → 1,80,38,602 (explicitly an estimate, not a tokenizer count)

## Reproduce the audit

Place the source JSONL at:

`data/raw/reasoning-summaries-61k.jsonl`

Then run:

```bash
python3 clean_dataset.py
```

The script uses only the Python standard library. It writes the clean corpus,
quarantine and `data/cleanup-report.json`.

## Build the submission

```bash
python3 build_site.py
python3 -m http.server 4174 --directory dist
```

The deployable `dist` directory contains only the page assets and aggregate
report. Raw, clean and quarantined records are deliberately excluded.
