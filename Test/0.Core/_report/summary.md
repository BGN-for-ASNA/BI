# Verification checks

| Test | Quantity | Reference | Criterion | Value | Result |
|---|---|---|---|---|---|
| `test_enum_matches_default` | mu | implicit enumeration (NumPyro) | max |Δ| < 1e-10 | 0 | ✅ |
| `test_enum_matches_default` | sigma | implicit enumeration (NumPyro) | max |Δ| < 1e-10 | 0 | ✅ |
| `test_enum_matches_default` | w | implicit enumeration (NumPyro) | max |Δ| < 1e-10 | 0 | ✅ |
| `test_enum_is_exact_enumeration` | mu | infer={"enumerate": "parallel"} | max |Δ| = 0 | 0 | ✅ |
| `test_enum_is_exact_enumeration` | sigma | infer={"enumerate": "parallel"} | max |Δ| = 0 | 0 | ✅ |
| `test_enum_is_exact_enumeration` | w | infer={"enumerate": "parallel"} | max |Δ| = 0 | 0 | ✅ |

![posterior comparisons](comparisons.png)

![z-scores](z_scores.png)
