"""Run:  python -m pytest -q test_model.py    (or  python test_model.py)"""
import math, numpy as np, pandas as pd
import data, model as M
from scipy.special import gamma

G, C = data.DEFAULT_GLOBALS, data.DEFAULT_COMPONENTS


def _short_lived_index(eng):
    """Index of the component whose MTTF is shortest, i.e. best resolved inside the
    15-year grid (limit tests need T >> MTTF)."""
    return int(eng.c.mu.idxmin())


def test_renewal_exponential_exact():
    dt, n = 50, 600
    Mf = M.renewal_function(1.0, 20000, dt, n); t = np.arange(n + 1) * dt
    assert np.max(np.abs(Mf[1:] - t[1:] / 20000) / (t[1:] / 20000)) < 1e-5

def test_renewal_weibull_asymptote():
    b, e, dt, n = 2.0, 20000, 50, 3000
    Mf = M.renewal_function(b, e, dt, n)
    mu = e * gamma(1 + 1 / b); var = e**2 * (gamma(1 + 2 / b) - gamma(1 + 1 / b)**2)
    asym = n * dt / mu + (var - mu**2) / (2 * mu**2)
    assert abs(Mf[-1] - asym) / asym < 1e-3

def test_block_to_rtf_limit():
    eng = M.Engine(C, G); i = _short_lived_index(eng)
    Tbig = eng.n * eng.dt
    r = eng.c.loc[i]
    blk = (r.Cp_ind + r.Cf * eng.M_at(i, Tbig)) / Tbig
    assert abs(blk - r.Cf / r.mu) / (r.Cf / r.mu) < 0.15      # block PM -> RTF as T grows

def test_age_policy_equals_rtf_for_infinite_T():
    eng = M.Engine(C, G); i = _short_lived_index(eng); r = eng.c.loc[i]
    age_big = (r.Cp_ind * eng.R[i][-1] + r.Cf * eng.F[i][-1]) / eng.I[i][-1]
    assert abs(age_big - r.Cf / r.mu) / (r.Cf / r.mu) < 0.12

def test_constant_hazard_never_pm():
    c = C.copy(); c.loc[:, "beta"] = 1.0; c["critical"] = False
    ind = M.Engine(c, G).individual_policies()
    assert (ind.policy == "Run-to-failure").all()

def test_decreasing_hazard_never_pm():
    c = C.copy(); c.loc[:, "beta"] = 0.7; c["critical"] = False
    ind = M.Engine(c, G).individual_policies()
    assert (ind.policy == "Run-to-failure").all()

def test_critical_part_never_rtf_and_respects_Rmin():
    res = M.run_pipeline(C, G); eng = res["engine"]
    plan, _ = M.group_plan(eng, res["individual"], res["group"]["recommended"])
    for i, r in plan.iterrows():
        if eng.c.critical[i] and eng.c.beta[i] > 1.05:
            assert r.k > 0
            assert eng.R_at(i, r.interval_mo * eng.u) >= G["r_min"] - 1e-6

def test_group_beats_or_equals_independent_block():
    g = M.run_pipeline(C, G)["group"]
    assert g["grouped_pm"] <= g["independent_block_pm"] * 1.0001

def test_grouping_gain_grows_with_setup_cost():
    lo = M.run_pipeline(C, {**G, "setup_fixed": 5000})["group"]["saving_vs_indep_block_pct"]
    hi = M.run_pipeline(C, {**G, "setup_fixed": 120000})["group"]["saving_vs_indep_block_pct"]
    assert hi > lo

def test_warranty_cost_monotone_in_W_and_has_interior_optimum():
    w = M.run_pipeline(C, G)["warranty"]["FRW (no PM)"]["curve"]
    assert (np.diff(w.wcost) >= -1e-6).all()
    best = w.loc[w[w.feasible].profit.idxmax()]
    assert G["w_min"] < best.W < G["w_max"]

def test_prw_cheaper_than_frw():
    eng = M.Engine(C, G)
    for W in (12, 24, 48):
        assert eng.warranty_point(W, "PRW")["cost"] <= eng.warranty_point(W, "FRW")["cost"] + 1e-9

def test_montecarlo_agrees():
    eng = M.Engine(C, G); v = eng.mc_validate(_short_lived_index(eng), 40, n=200000, seed=7)
    assert abs(v["M_err_pct"]) < 2.0 and abs(v["age_err_pct"]) < 1.0

def test_k_max_is_honoured():
    ev = M.Engine(C, {**G, "k_max": 4}).eval_group(6.0)
    assert ev["table"].k.max() <= 4

def test_horizon_reason_is_distinct():
    c = C.copy(); c["critical"] = False; c.loc[:, "beta"] = 3.0; c.loc[:, "eta_h"] = 5e6
    ind = M.Engine(c, G).individual_policies()
    assert ind.reason.str.contains("planning horizon").all()

def test_warranty_schedule_basis_switch():
    a = M.run_pipeline(C, G, "recommended"); b = M.run_pipeline(C, G, "best")
    assert a["schedule_basis"] == "recommended" and b["schedule_basis"] == "best"

def test_export_workbook_roundtrip():
    import io, zipfile
    res = M.run_pipeline(C, G)
    tbl = M.export_tables(res, G, {"Failure modes": data.FAILURE_MODES})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in tbl.items():
            df.to_excel(xw, sheet_name=name, index=False)
    back = pd.read_excel(io.BytesIO(buf.getvalue()), sheet_name=None)
    assert set(back) == set(tbl)
    comp = back["Components"]
    assert np.allclose(comp["Cf"], res["engine"].c["Cf"], rtol=1e-10)
    assert "source" in comp.columns and comp["source"].notna().all()
    assert all(len(k) <= 31 for k in tbl)

def test_data_integrity():
    assert {"source"} <= set(data.DEFAULT_COMPONENTS.columns) and "source" in data.FAILURE_MODES.columns
    keys = set(data.REFERENCES)
    for cell in list(data.DEFAULT_COMPONENTS.source) + list(data.FAILURE_MODES.source):
        for k in (x.strip() for x in cell.split(",")):
            assert k in keys or k == "Baseline", k
    c = data.DEFAULT_COMPONENTS
    assert c.eta_h.notna().all() and (c.eta_h > 0).all()
    # L10 -> eta conversion reproduces the reference life
    r = c[c.life_p.notna()].iloc[0]
    L = r.eta_h * (-math.log(1 - r.life_p)) ** (1 / r.beta)
    assert abs(L - r.life_ref_h) / r.life_ref_h < 1e-4

def test_extreme_inputs_do_not_crash():
    for gg in ({"hours_per_month": 150.0}, {"r_min": 0.999}, {"a_min": 0.999}, {"setup_fixed": 0, "setup_hours": 0},
               {"downtime_cost": 0}, {"w_min": 1.0, "w_max": 96.0}, {"t_min_months": 6.0}, {"budget_per_year": 1e3}):
        res = M.run_pipeline(C, {**G, **gg})
        assert np.isfinite(res["group"]["best"]["cost_pm"])
    c = C.copy(); c["beta"] = 5.0; c["eta_h"] = 5000
    assert np.isfinite(M.run_pipeline(c, G)["group"]["best"]["cost_pm"])

if __name__ == "__main__":
    import sys, inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print("PASS", name)
            except Exception as e: fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
