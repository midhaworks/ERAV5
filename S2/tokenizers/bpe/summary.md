# BPE summary

## Primary strategy: evaluate every fourth language

The main strategy was to measure the fourth-language choice rather than assume
it. The vocabulary corpus consists of faithful Markdown from Wikipedia's
**India** page in each language. English, Hindi, and Telugu were fixed, then
every one of the 306 other cached Wikipedia “India” pages was trained as the
fourth language. Each candidate received the same 16 declared training-weight
profiles in a sequential exhaustive run. Only after all 306 candidates were
fully optimized was the highest reproducible score promoted.

This matters because the earlier baseline-only shortlist missed Sindhi. The
exhaustive profile search found Sindhi with the top score, well ahead of
Italian and Persian.

## Idea

This is one shared Hugging Face BPE tokenizer with a Metaspace pre-tokenizer
and decoder. It uses no Unicode normalizer, so distinct input strings are not
silently rewritten. A single `[UNK]` special token is included in the exact
10,000-entry vocabulary.

## Process

- Fix English, Hindi, and Telugu Wikipedia “India” pages.
- Test every one of 306 available Wikipedia languages as the fourth page.
- Train every candidate with the same baseline policy, then evaluate all 16
  declared language-weight profiles for every candidate sequentially.
- Select the maximum adjusted score, promote its exact four-page corpus, and
  rebuild the tokenizer independently.
- Test complete pages, language samples, the rejection sentence, and 200 random
  substrings.

## Outcome

- Languages: English, Hindi, Telugu, and Sindhi.
- Training weights: 3, 4, 6, and 4 respectively.
- Exact vocabulary size: 10,000.
- Adjusted score: 86,311.02.
- Fertility spread: 0.01158601.
- Zero unknown tokens and exact round trips on all four full pages.
- The rejection sentence and all 204 independent probes pass.

“BPE” is the correct concise name. “Shared Hugging Face BPE with Metaspace” is
the more precise technical description used in the review page.

## Possible improvements

1. **Minimum content benchmark:** predeclare a minimum number of word runs,
   faithful units, or characters for an eligible language page.
2. **Page-quality validation:** reject stubs, disambiguation pages,
   navigation-heavy pages, and pages without enough visible article content.
3. **Translation and topic comparability:** use translation-assisted semantic
   checks to determine whether localized India pages cover comparable concepts
   and sections.
4. **Report both views:** retain the complete unfiltered ranking and publish a
   second eligibility-filtered ranking so exclusions remain transparent and
   reproducible.
5. **Test threshold sensitivity:** use the review page's interactive filter to
   verify how eligibility changes. Sindhi remains the top-scoring BPE language at the shown
   0, 5,000, 7,560, and 10,000 visible-word thresholds.

## References

1. [ERA V5 Assignment 2 Reference Solution from Admin](https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmrirwdhc0afw08nmp64282xo/lesson)
2. [Layout inspiration — submission from a peer](https://dcsuiova318m4.cloudfront.net)
3. [MediaWiki REST API documentation](https://www.mediawiki.org/wiki/API:REST_API)
4. [Hugging Face Tokenizers components — BPE and Metaspace](https://huggingface.co/docs/tokenizers/main/components)

## Acknowledgements

Intern who helped code: Codex :)

---

Copyright 2026 Avnish Midha. All rights reserved.  
Author: Avnish Midha  
GitHub: [avnishbm](https://github.com/avnishbm)
