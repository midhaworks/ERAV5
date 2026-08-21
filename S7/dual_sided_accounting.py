"""Generated parameter accounting for vocabulary-independent input/output paths."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent / "artifacts" / "dual_sided_accounting"


def report(vocab: int = 131_072, model_dim: int = 4096, code_states: int = 258,
           code_dim: int = 32) -> dict:
    standard_matrix = vocab * model_dim
    # The exact K-code is fixed (one-hot byte/position coordinates), so it has
    # zero trainable codebook parameters.  A single shared projection is used
    # for both input and output; the output produces byte/EOS/CONT states.
    fixed_codebook_parameters = 0
    kronecker_dimension = code_states * code_dim
    shared_projection = model_dim * kronecker_dimension
    untied_k_path = 2 * shared_projection
    untied_removed = 2 * standard_matrix - untied_k_path
    tied_removed = standard_matrix - shared_projection
    return {
        "inputs": {"vocab_size": vocab, "model_dim": model_dim,
                   "code_states": code_states, "code_dim": code_dim},
        "baseline": {"untied_input_parameters": standard_matrix,
                      "untied_output_parameters": standard_matrix,
                      "tied_shared_parameters": standard_matrix},
        "kronecker": {"fixed_codebook_parameters": fixed_codebook_parameters,
                       "kronecker_dimension": kronecker_dimension,
                       "shared_projection_parameters": shared_projection,
                       "untied_parameters_for_two_sides": untied_k_path,
                       "tied_parameters_for_two_sides": shared_projection,
                       "vocab_dependent_parameters": 0},
        "savings": {"untied_input_plus_output_parameters_removed": untied_removed,
                    "untied_reduction_fraction": untied_removed / (2 * standard_matrix),
                    "tied_parameters_removed": tied_removed,
                    "tied_reduction_fraction": tied_removed / standard_matrix},
        "interpretation": "Dual-sided savings approximately double only against an untied baseline; tied baselines have one shared Vxd matrix.",
    }


def sweep() -> list[dict]:
    return [{"vocab_size": v, **report(vocab=v)["savings"],
             "standard_matrix_parameters": report(vocab=v)["baseline"]["tied_shared_parameters"],
             "shared_k_projection_parameters": report(vocab=v)["kronecker"]["shared_projection_parameters"]}
            for v in (32_000, 131_072, 1_000_000)]


if __name__ == "__main__":
    result = report(); result["vocabulary_sweep"] = sweep(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
