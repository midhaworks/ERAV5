# BPE — 5,000-visible-word submission

This folder contains a genuine shared Hugging Face BPE approach. English,
Hindi, and Telugu are fixed; every other cached Wikipedia “India” page is tested
as the fourth language. With the declared 5,000-visible-word minimum and no
writing-direction filter, **Sindhi** is the selected fourth language.

The candidate search reads the shared cache at
`../piecevocab/candidate-corpus/` by default, avoiding a duplicate copy.
The final winning four-page corpus is copied into this folder after selection.

Key design choices:

- one shared BPE tokenizer, exactly 10,000 entries including `[UNK]`
- no Unicode normalizer
- Metaspace pre-tokenizer and decoder
- `min_frequency=1`
- faithful-unit denominator matching `S2-reference`
- fixed baseline weights `(English=3, Hindi=4, Telugu=4, fourth=2)` for every
  candidate, followed by the same declared finite profile search for every
  candidate by default

The submitted Hugging Face artifact is
`search-visible-5000/winner.tokenizer.json`; `tokenizer.json` is the identical
promoted copy used by the standalone evaluator.

```bash
python3 tokenizers/bpe/search_fourth_languages.py --optimize-top 0
python3 prepare_visible_selection.py
python3 tokenizers/bpe/train_tokenizer.py
python3 tokenizers/bpe/evaluate_tokenizer.py
```

Use `--reuse-baseline` to resume from `search-results/ranking.json` and the
per-candidate `profile-checkpoint.json`. The search is sequential (no Python
multiprocessing), and exhaustive runs checkpoint every completed candidate.

Files:

- `bpe_common.py` — tokenizer construction and faithful evaluator
- `search_fourth_languages.py` — all-candidate baseline and finalist search
- `search-results/ranking.json` — complete 306-candidate ranking
- `search-visible-5000/ranking.json` — the 86 eligible candidates
- `search-visible-5000/winner.tokenizer.json` — final Sindhi tokenizer
- `train_tokenizer.py` — direct rebuild from the selected four pages
- `evaluate_tokenizer.py` — exact page/sample/substrings and score checks
- `corpus/` — exact final four-page snapshots
- `tokenizer.json` and `metrics.json` — final submission artifacts
