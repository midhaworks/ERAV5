# PieceVocab content-filtered selection summary

- Source fourth-language candidates: 306
- Writing-direction exclusions: 0
- Minimum visible word runs: 5,000
- Final eligible candidates: 86
- Selected languages: English, Hindi, Telugu, and Yoruba
- Exact vocabulary size: 10,000
- Adjusted score: 7,86,647.20
- Fertility spread: 0.00127122
- All four complete pages round-trip exactly

The complete 306-language search remains the source ranking. Unicode RTL text is
stored in logical character order, so writing direction is not an eligibility
rule. The tokenizer consumes the serialized sequence normally.

The 5,000-visible-word content rule is applied before winner selection. It is
tokenizer-independent and leaves 86 candidates. A generated-token cutoff would
be circular and would create different eligibility sets for PieceVocab and BPE.
The PieceVocab result is threshold-sensitive: Pashto becomes the winner at
6,000, so the interactive report preserves both views.

Without the visible-word filter, Yiddish is the PieceVocab winner with a score
of 18,33,650.92. Its 2,404 visible word runs—not its RTL direction—exclude it
from the submitted 5,000-run result.
