# Nurse Scheduling Problem under the Trustworthy AI Lens

A comparative evaluation of three approaches to the **Nurse Scheduling Problem** — exact MILP,
simulated annealing, and an ML + MILP hybrid — against four attributes of trustworthy AI:
**critical-constraint satisfaction, fairness, robustness, and explainability**.

Interdisciplinary project ("Projet Fil Rouge") of the *Trustworthy AI* Advanced Master's programme,
CentraleSupélec — Université Paris-Saclay, 2026.

> The full report is included in this repository and is written in French. Section, table and
> figure numbers referenced below point to that report.

---

## Motivation

Staff scheduling in healthcare is classified as **high-risk** under the EU AI Act (Annex III, 4b):
it encodes statutory rules — inter-shift rest, weekly working time, minimum coverage — and its output
affects both patient safety and staff workload. It is also an ideal methodological testbed:
combinatorial, multi-objective, hard-constrained, and **free of any sensitive clinical data**.

The question addressed here is not *which method is fastest*, but **what "trustworthy" means for each
method, attribute by attribute**.

## What this repository contains

| Approach | Principle | Position on the admissibility guarantee |
|---|---|---|
| **MILP** (Gurobi) | exact model: 6 hard constraints, 4 weighted penalties | **formal** guarantee by construction, with a certified suboptimality bound |
| **Simulated annealing** | dense N×D encoding, energy `E = Z + λ·V` | no guarantee — admissibility is **statistical** |
| **ML + MILP hybrid** | XGBoost predicts an initial solution, passed as a `MIPStart` | guarantee **inherited** from the solver, plus local explainability (SHAP) |

The model and metrics are specified in §4 of the report; headline results are summarised below.

## Layout

```
NSP_fil_rouge.ipynb        consolidated notebook, runnable end to end
  §1  Model                instance, MILP solver (cold / warm), ML features
  §2  Dataset              50 instances, MILP labels (180 s, 3 % gap), 40/10 split
  §3  Simulated annealing  encoding, energy, neighbourhoods, cooling schedule
  §4  Experiment 1         MILP vs SA at equal budget (300 s)              -> Table 4
  §5  Training             XGBoost + MLP (seed 42)
  §6  Experiment 2         cold / warm-XGB / warm-MLP trajectories, 5→60 s -> Figure 1
  §7  Experiment 3         stress test over N (D = 28 fixed), N = 24 → 17
  §8  Experiment 4         2×2 factorial design (strict 11 h rest × night-dispersion
                           penalty) across all three approaches — added after submission
```

Every expensive step is cached under `RESULTS_DIR`; re-running only recomputes what is missing.
Set `FORCE_RERUN = True` to recompute everything.

## Requirements

- Python ≥ 3.10 with `gurobipy`, `xgboost`, `torch`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`
- **A Gurobi licence** able to handle models of roughly 2,900 binary variables. The size-limited
  licence bundled with `pip install gurobipy` is **not** sufficient (2,000-variable cap).

Licence credentials are read from the environment — **no secrets are committed**:

```bash
export GRB_WLSACCESSID="..."
export GRB_WLSSECRET="..."
export GRB_LICENSEID="..."
export NSP_RESULTS="./results"      # optional, defaults to ./results
```

## Runtimes (first run, single core)

| Section | Cost | Cache file |
|---|---|---|
| §2 Dataset (50 × MILP at 180 s) | ≈ 2 h | `dataset_raw.pkl`, `dataset_features.parquet` |
| §4 MILP vs SA (10 instances × (1 MILP + 10 SA seeds)) | ≈ 8 h | `table4_milp_sa.pkl` |
| §5 Training | ≈ 10 min | `xgb_model.pkl`, `nn_model.pt`, `nn_scaler.pkl` |
| §6 Trajectories (3 × 10 instances × 60 s) | ≈ 35 min | `trajectories_5_60.npz` |
| §7 Stress test over N | ≈ 40 min | `stress_test_N.csv` |
| §8 Factorial design (48 MILP runs + 20 SA runs) | ≈ 4.4 h | `factorial_2x2_repl.pkl` |

## Key findings

**At equal budget (300 s), no approach Pareto-dominates the others** across the four attributes.
The MILP wins on the objective and on preference satisfaction, guarantees admissibility by
construction, and repairs a disruption in under a second (versus ~157 s for simulated annealing);
SA retains an edge in time-to-first-solution and in potential scalability.

**Warm-starting yields a purely primal gain.** It makes reaching admissibility reliable under a tight
budget — 10/10 instances admissible by ~10 s, versus ~30 s cold — but the trajectories converge beyond
~45 s: final solution quality is unchanged. XGBoost and MLP trajectories overlap, so the benefit comes
from the warm-start paradigm rather than from the specific model.

**Optimality is only guaranteed relative to the chosen formalisation.** The optimal schedule contains
blocks of consecutive night shifts — a direct consequence of a continuity penalty applied uniformly
across shift types. The §8 factorial design (3 replicates, `Threads=1`, differences paired by seed)
quantifies the two candidate remedies:

| Variant | Cost on Z (median, [min, max]) | Night blocks (median [min, max]) |
|---|---|---|
| none (baseline) | — | 41 [39, 44] |
| C1 — night-dispersion penalty | +6.64 % [5.95, 6.86] | **1 [0, 2]** |
| C2 — strict 11 h statutory rest | +0.68 % [0.19, 0.99] | 43 [41, 43] |
| C3 — both | +7.16 % [6.97, 7.37] | **3 [2, 3]** |

The **interaction is negligible** (−0.16 pt): the two remedies add up without interfering. Only the
dispersion penalty removes the night blocks; the 11 h rule alone does not contribute. Measured noise
floor: 1.0 pt on Z (0.42 %) for the MILP under an identical configuration.

## Known issues and post-submission corrections

The following were identified **after the report was submitted**, while consolidating the code and
replicating the experiments. They are listed here for traceability.

1. **Two divergent implementations of `P_cont`.** The MILP counts *leaving a work shift*
   (`c[n,d] ≥ x[n,d,s] − x[n,d+1,s]`, s ∈ {M,A,N}); the SA counted *two consecutive worked days*.
   The two diverge most sharply on same-shift blocks: an N-N-N-N block costs 1 under the MILP metric
   and 3 under the SA metric. The Table 4 harness recorded **each method's own penalties**, so the
   `P_cont` row is not comparable and the SA's reported continuity advantage **is not supported**.
   Re-evaluated with a single shared evaluator on the reference instance, SA is in fact *worse* than
   the MILP on that dimension. The MILP's dominance on Z is unaffected (in fact strengthened).
   §8 uses one shared evaluator across all three approaches.

2. **"Strict rest reinforces night blocks" (§6.5) does not replicate.** That claim rested on a single
   run. Across 3 replicates with `Threads=1`: baseline = 41 [39, 44] versus C2 = 43 [41, 43] — the
   intervals overlap, so the effect sits below solver noise.

3. **The argmax projection only guarantees H1** — stated in §5.3 of the report, but its magnitude was
   not measured: the ML prediction violates coverage heavily (~165 H2 violations on the reference
   instance, comparable on the test instances). The gain observed in Figure 1 therefore comes from the
   **solver repairing the MIP start**, not from a start accepted as-is. Suggested fix: a *Req-aware*
   projection — for each (day, shift), select the `Req(d,s)` nurses with the highest predicted
   probability — which would guarantee H2 alongside H1.

4. **Solver non-determinism.** With multiple threads, three identical runs of the same model returned
   Z = 238.57 / 239.11 / 237.77 — noise larger than some of the effects under study. Fine-grained
   comparisons require `Threads = 1` and differences paired by seed.

## Licence

MIT — see [`LICENSE`](LICENSE). Free to reuse, including for research; a citation or acknowledgement
is appreciated if this code contributes to published work.

## Citation

```bibtex
@misc{gallo2026nsp,
  author = {Gallo, Pietro Marco},
  title  = {Hospital Scheduling under the Trustworthy-AI Lens:
            the Nurse Scheduling Problem},
  year   = {2026},
  note   = {Interdisciplinary project, Advanced Master's in Trustworthy AI,
            CentraleSupélec — Université Paris-Saclay},
  url    = {https://github.com/pietromarcogallo/NSP_trustworthy_AI_interdisciplinary_project/}
}
```
