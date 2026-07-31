# PieceVocab — 5,000-visible-word submission

PieceVocab is the custom reversible tokenizer used by the final S2 review. The
fixed languages are English, Hindi and Telugu; after applying the declared
minimum of 5,000 visible Unicode word runs to all 306 candidates, **Yoruba** is
the selected fourth language.

The submitted tokenizer is:

```text
search-visible-5000/winner.tokenizer.json
```

`piecevocab.py` is its required executable definition. The JSON is not a
standalone Hugging Face tokenizer, so evaluators must load both files:

```python
from piecevocab import load

tokenizer = load("search-visible-5000/winner.tokenizer.json")
text = "India's population is 1,428,627,663."
assert tokenizer.decode(tokenizer.encode(text)) == text
```

## Reproduce

From the repository's `S2/` directory:

```bash
python3 prepare_visible_selection.py
python3 tokenizers/piecevocab/train_tokenizer.py
python3 tokenizers/piecevocab/evaluate_tokenizer.py
```

Important files:

- `piecevocab.py` — encoder, decoder and universal Unicode fallback
- `candidate-corpus/` — cached faithful Wikipedia pages used by selection
- `search-results/ranking.json` — complete unfiltered 306-language ranking
- `search-visible-5000/ranking.json` — the 86 eligible candidates
- `search-visible-5000/winner.tokenizer.json` — final Yoruba tokenizer
- `rank_fourth_languages.py` — exhaustive candidate ranking
- `download_all_languages.py` — reproducible MediaWiki corpus acquisition
- `train_tokenizer.py` and `evaluate_tokenizer.py` — final rebuild and checks

Writing direction is not filtered. Unicode right-to-left text is serialized in
logical order before tokenization, so removing RTL languages would be incorrect.
