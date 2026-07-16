# ProteinStabilizer training report

Date: 2026-07-16
Seed: `20260715`

## Outcome

The application now has two deliberately separate inference stages:

- ESM-C 600M remains the fast default GPCR saturation screen and the only
  practical first-pass model.
- A full ESM-C 6B single-mutant ensemble supplies a stronger general ddG,
  improved ProTherm/MPTherm transfer heads, and an optional bounded GPCR
  reranker.
- A separate full-6B permutation-invariant head now supplies the preferred
  learned correction for double mutants.

Solubility and structure branches remain out of scope because their audited
transfer candidates did not improve new-receptor stability ranking.

## ESM-C 6B artifacts

- Runtime: Python 3.12, Transformers 4.57.6, ESM 3.3.0,
  PyTorch 2.9.1+cu128.
- Encoder: `biohub/ESMC-6B`, 2,560-dimensional final-layer residue vectors.
- Checkpoint-index SHA-256:
  `6846456e20e6ee2c37461f7bfc21d316d69bdaf165b925691afcb39e583244da`.
- Megascale cache: 136,466 sequences, 143,924 sites, 766.68 seconds.
- Megascale double cache: 125,823 sequences, 240,431 sites, 697.84 seconds.
- Focused transfer/GPCR cache: 7,502 sequences, 11,089 sites, 242.24 seconds.
- Megascale request manifest:
  `e69cd477a7113dc5447c354cbf0dcff8cc1b5f6f9ed2f4c8b36fef7f1bffb99f`.
- Focused request manifest:
  `f812e6e4e8e43c3131f836806cdef0491e7b573263b920cbb5be148fce72de37`.
- Megascale double request manifest:
  `b6127258998b8599a4db1e16609100c299938294b25defe9c2050c91dee79007`.

## Held-out results

| Evaluation | 600M Spearman | 6B Spearman | 600M MAE | 6B MAE |
| --- | ---: | ---: | ---: | ---: |
| Megascale single protein holdout | 0.777 | 0.818 | 0.556 | 0.516 |
| ProTherm protein holdout | 0.409 | 0.481 | 1.168 | 1.133 |
| MPTherm protein holdout | 0.222 | 0.274 | 3.652 | 3.517 |

The 6B ProTherm head also reaches Spearman `0.571`, MAE `0.968`, and RMSE
`1.368` on external S669.

On the 18,574-row protein-held-out Megascale-D double-mutant test, the deployed
6B learned estimate reaches Spearman `0.623`, Pearson `0.592`, MAE `0.741`, and
RMSE `0.983`. Its corresponding 6B additive baseline reaches Spearman `0.615`,
MAE `1.142`, and RMSE `1.468`. The existing 600M learned estimate reaches
Spearman `0.545`, MAE `0.851`, and RMSE `1.144`.

## GPCR application decision

The fast default rank remains:

```text
0.80 * 600M MPTherm percentile
+ 0.20 * 600M masked-marginal percentile
```

The optional second-stage 6B rank is:

```text
0.60 * 600M MPTherm percentile
+ 0.15 * 600M masked-marginal percentile
+ 0.25 * 6B MPTherm percentile
```

The 6B share was selected only inside a conservative `0.00`–`0.25`
development grid. GPCR-tm development macro within-receptor Spearman improves
from `0.086` to `0.117`; four receptors improve, six tie, and none worsen. The
two evaluable official-test receptors improve from macro `0.800` to `1.000`,
but the 12-row set has been consulted by earlier stages and is confirmatory,
not untouched.

On the independent C5aR scan, the optional rank improves AUC from `0.678` to
`0.696`, average precision from `0.255` to `0.317`, and positives in the top
50 from 13 to 15. The paired-bootstrap 95% interval for the AP gain is
`+0.005` to `+0.116`.

## Checkpoint integrity

- `single_ensemble.pt`:
  `e227ed23a215f7355a705ab51a1cd3c81ae0b44cfdb4fd07154d956b8cf518fe`
- `single_head.pt`:
  `67a34284e98aa20cb1f595645f3bf11f7c9a4b04a304fea1a4fa1c1c4dd6f017`
- `protherm_ddg_head.pt`:
  `ef241664fe8a2b28d40114e6f747751bf8f2ee10fdc29f7b816c017e8dc57b49`
- `mptherm_dtm_head.pt`:
  `90fa914103cb94e08bc2cd968e90f05fa97c6516084ac157511bf95015b7867d`
- `multi_head.pt`:
  `1a94cb8ed93d0891614007092895498dd5a6ce18b17670972e535ddb17760b18`

## Interpretation

ESM-C 6B is a material generic stability improvement and is ready for
single-mutant application. It is not a standalone GPCR solution: the 6B-only
rank failed the small receptor test, so the validated 600M signal stays in the
optional blend. Full-6B multiple-mutant inference is now available and reports
the ensemble additive and learned epistasis components separately. Its evidence
is protein-disjoint Megascale-D double-mutant performance, not direct GPCR
combination data; sets larger than two mutations remain extrapolations.
