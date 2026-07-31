# S2 — Multilingual tokenizer assignment

This dependency-free static site reviews both final submissions and runs their
saved tokenizers in the browser.

The published selection policy keeps Wikipedia “India” page languages with at
least 5,000 visible Unicode word runs after Markdown URL destinations are
removed. Writing direction is not filtered: Unicode RTL text remains serialized
in logical order for tokenization. This leaves 86 of the 306 evaluated
fourth-language candidates. Yoruba wins for PieceVocab; Sindhi wins for BPE.

The browser keeps all 306 ranking rows available for threshold sensitivity
analysis. The interactive table defaults to the selected 5,000-run policy and
can be changed without altering the submitted tokenizer artifacts. Source
content—not generated token count—is used for eligibility so the filter remains
independent of tokenizer approach and script fertility.

```bash
python3 prepare_visible_selection.py
python3 build_site.py
python3 -m http.server 8000 --directory dist
```

For Netlify, create a site with `S2` as the base directory. The included
`netlify.toml` builds to `dist/`.

## Repository structure

```text
S2/
├── README.md
├── site/                         # editable review-site source
├── dist/                         # deployable static site
├── artifacts/                    # submission summaries
├── tokenizers/
│   ├── piecevocab/               # custom reversible tokenizer + rankings
│   └── bpe/                      # Hugging Face BPE + rankings
├── prepare_visible_selection.py  # applies the 5,000-word policy
├── build_site.py                 # packages the review site
└── netlify.toml
```

Superseded experiments are retained locally in `archive/` and intentionally
excluded from Git. Virtual environments and generated Python caches are also
excluded.
