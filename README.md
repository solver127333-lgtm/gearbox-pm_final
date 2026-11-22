# Industrial Helical Gearbox — Preventive Maintenance & Warranty Optimizer

A standalone Streamlit application that selects maintenance policies, builds a shared-shutdown
maintenance calendar and optimises warranty length for a two-stage helical industrial gearbox
(~75 kW, 1500 rpm input, continuous conveyor duty, 600 operating hours per month).

## Run

    pip install -r requirements.txt
    streamlit run app.py          # local
    python test_model.py          # 18 automated checks (or: python -m pytest -q test_model.py)

Deployment on Streamlit Community Cloud: point the app at `app.py`; `requirements.txt` is the only
dependency file needed (Python 3.10+; tested with pandas 3.x, numpy 2.x, streamlit 1.x).

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit application (10 tabs, in-app Excel / CSV / JSON export) |
| `model.py` | Weibull reliability, renewal equation, RTF / age / time-based PM, group PM, warranty, Monte-Carlo, sensitivity, export tables |
| `data.py` | Cited component, failure-mode and baseline data; `REFERENCES` bibliography |
| `test_model.py` | Mathematical-invariant, robustness and export tests |
| `requirements.txt` | Dependencies |

## Application tabs

Overview · System & Data · Warranty Optimization · PM Policies · Group PM · Constraints & Maintainability ·
Results Analysis · Calculation Audit (step-by-step derivation of every number) · Methodology & References · Export.

## Method

- Weibull: `F(t)=1-exp[-(t/η)^β]`, `μ=η·Γ(1+1/β)`; lives quoted at failure probability p convert by
  `η = L_p / [-ln(1-p)]^(1/β)` (for L10: `η = L10 / 0.1054^(1/β)`).
- Renewal function `M(t)=F(t)+∫M(t-x)dF(x)`, second-order trapezoid scheme on a 15-year grid.
- Cost rates per operating hour: run-to-failure `Cf/μ`; age-based `[Cp·R(T)+Cf·F(T)]/∫R`; time-based `[Cp+Cf·M(T)]/T`.
- Group PM: base interval T0, each component on a multiple k of T0 (k ≤ `k_max`) or run to failure; exact
  shared-visit frequency from the least common multiple of the chosen k; coordinate-descent refinement.
  Constraints: R ≥ R_min for safety-critical wear-out parts, availability, annual budget, shortest interval.
- Warranty: free-replacement (FRW) and pro-rata (PRW); optional PM-schedule scenario that follows either the
  recommended calendar schedule or the global optimum (sidebar switch). Profit `Π(W)=P0+ΔP(W)−Cprod−E[Cw(W)]`.

## Modelling assumptions

1. **Sequential single crew (conservative):** components sharing a shutdown are replaced one after another,
   so planned repair hours `t_p` are summed within a maintenance group, in addition to the shared shutdown hours.
2. The shared visit's fixed cost and shutdown hours are paid once per visit.
3. Replacement and corrective repair restore as-good-as-new; components fail independently.
4. Two-parameter Weibull (no failure-free interval): conservative for gears and shafts.
5. 15-year planning horizon. If the cost-optimal PM interval of an unconstrained wear-out part exceeds it,
   the part runs to failure and the stated reason says so (distinct from "no interval beats run-to-failure").
6. Costs in INR; no discounting.

## Data provenance

`data.py` separates three classes of input (each row of `DEFAULT_COMPONENTS` carries `life_basis` and `source`):

1. **Literature-derived reliability parameters.** Weibull shapes from Beermann (2018, after Bertsche) and Zaretsky
   (2016); bearing reference lives from the worked 66 kW helical-unit example of Gabelli et al. (2015); gear and
   shaft rated lives from AGMA/ISO rating bases (Rinaldo 2018); oil, seal and breather intervals from Flender and
   SEW-EURODRIVE operating instructions. Scale `η` is computed with the stated conversion.
2. **Engineering baselines** where no numeric public value was found: Weibull shapes for oil, seals, breather and
   housing, repair durations `t_p`/`t_f`, part and collateral costs. OREDA and NPRD-2016 databases were not
   accessible for this release; they are the recommended calibration sources for these rows.
3. **Configurable Site / Commercial Baseline Parameters** (downtime cost, labour rate, visit cost, budget, warranty
   price curve): plant- and market-specific, listed with typical ranges in the application.

Replace class 2 and 3 values with plant records and supplier quotations before using results for a decision.
All values are editable in the application; full citations are in the *Methodology & References* tab and in
`data.REFERENCES`.
