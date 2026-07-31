# PieceVocab submission instructions

## Required files

- `piecevocab.py` — executable tokenizer with `encode()` and `decode()`
- `piecevocab.tokenizer.json` — exact 10,000-entry vocabulary and metadata

Load the tokenizer with:

```python
from piecevocab import load

tokenizer = load("piecevocab.tokenizer.json")
assert tokenizer.decode(tokenizer.encode("Lossless text 🧪")) == "Lossless text 🧪"
```

## Selection policy

English, Hindi, and Telugu are fixed. The fourth language was selected from
all 306 Wikipedia “India” pages regardless of writing direction. Eligible pages
contain at least 5,000 visible Unicode word runs after Markdown URL destinations
are removed.

This pre-tokenizer content rule leaves 86 eligible candidates. Generated token
counts are not used as an eligibility filter because they depend on the
candidate tokenizer and would make selection circular. The selected fourth
language is Yoruba (`yo`).
