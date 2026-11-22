"""
app.py -- Industrial Helical Gearbox: Preventive Maintenance & Warranty Optimizer
Run:   streamlit run app.py
"""
import io, json, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as D
import model as M

st.set_page_config(page_title="Gearbox PM & Warranty Optimizer", layout="wide", page_icon="⚙️")
BLUE, ORANGE, GREEN, RED, GREY = "#0B3C7D", "#E07B00", "#2E8B57", "#C0392B", "#7F8C8D"


# ------------------------------------------------------------------ cached solvers
@st.cache_resource(show_spinner="Solving renewal equations and optimising...")
def compute(comps_json: str, g_json: str, basis: str):
    comps = pd.read_json(io.StringIO(comps_json), orient="split")
    g = json.loads(g_json)
    res = M.run_pipeline(comps, g, schedule_basis=basis)
    res["plan_best"] = M.group_plan(res["engine"], res["individual"], res["group"]["best"])
    res["plan_rec"] = M.group_plan(res["engine"], res["individual"], res["group"]["recommended"])
    return res


@st.cache_data(show_spinner="Running one-at-a-time sensitivity...")
def cached_sens(comps_json, g_json, delta):
    comps = pd.read_json(io.StringIO(comps_json), orient="split")
    return M.sensitivity(comps, json.loads(g_json), delta)


@st.cache_data(show_spinner="Sweeping constraints...")
def cached_sweeps(comps_json, g_json):
    comps = pd.read_json(io.StringIO(comps_json), orient="split")
    return M.constraint_sweeps(comps, json.loads(g_json))


def inr(x):
    return f"Rs {x:,.0f}"


# ------------------------------------------------------------------ sidebar : baseline parameters
g = dict(D.DEFAULT_GLOBALS)
with st.sidebar:
    st.title("⚙️ Parameters")
    st.caption("Configurable Site / Commercial Baseline Parameters. Results update automatically. "
               "Typical ranges are listed in the *System & Data* tab.")
    with st.expander("Operating profile & cost drivers", expanded=True):
        g["hours_per_month"] = st.number_input("Operating hours / month", 100.0, 730.0, g["hours_per_month"], 10.0)
        g["labour_rate"] = st.number_input("Labour rate (Rs/h)", 0.0, 20000.0, g["labour_rate"], 100.0)
        g["downtime_cost"] = st.number_input("Downtime cost (Rs/h)", 0.0, 200000.0, g["downtime_cost"], 1000.0)
        g["setup_fixed"] = st.number_input("Fixed cost per maintenance visit (Rs)", 0.0, 500000.0, g["setup_fixed"], 1000.0,
                                           help="Crane, scaffolding, lock-out, oil handling, travel; shared when parts are grouped")
        g["setup_hours"] = st.number_input("Shutdown hours per visit", 0.0, 24.0, g["setup_hours"], 0.5)
    with st.expander("Constraints"):
        g["r_min"] = st.slider("Min reliability at replacement (safety-critical parts)", 0.50, 0.999, g["r_min"], 0.005)
        g["a_min"] = st.slider("Min system availability", 0.95, 0.999, g["a_min"], 0.001, format="%.3f")
        g["budget_per_year"] = st.number_input("Max maintenance budget (Rs/year)", 100000.0, 5e6, g["budget_per_year"], 50000.0)
        g["pm_saving_threshold"] = st.slider("PM must save at least (vs run-to-failure)", 0.0, 0.20, g["pm_saving_threshold"], 0.005)
        g["t_min_months"] = st.number_input("Shortest practical PM interval (months)", 0.5, 12.0, g["t_min_months"], 0.5)
        g["k_max"] = st.select_slider("Largest multiple of base interval (k_max)", options=list(M.KSET), value=g["k_max"])
    with st.expander("Warranty & commercial"):
        g["base_price"] = st.number_input("Base price (Rs)", 1e5, 1e7, g["base_price"], 10000.0)
        g["unit_cost"] = st.number_input("Production cost per unit (Rs)", 1e5, 1e7, g["unit_cost"], 10000.0)
        g["premium_cap"] = st.number_input("Max price premium for long warranty (Rs)", 0.0, 1e6, g["premium_cap"], 5000.0)
        g["premium_tau"] = st.number_input("Premium saturation time (months)", 1.0, 120.0, g["premium_tau"], 1.0)
        g["w_min"], g["w_max"] = st.slider("Allowed warranty length (months)", 1.0, 120.0, (g["w_min"], g["w_max"]), 1.0)
        g["warranty_budget_frac"] = st.slider("Max warranty cost (% of price)", 0.01, 0.30, g["warranty_budget_frac"], 0.01)
        g["max_claims"] = st.slider("Max expected claims per unit", 0.05, 3.0, g["max_claims"], 0.05)
        g["warranty_covers_downtime"] = st.checkbox("Warranty compensates customer downtime", g["warranty_covers_downtime"])
        g["pm_paid_by_maker"] = st.checkbox("Manufacturer pays planned PM during warranty", g["pm_paid_by_maker"])
    basis_label = st.radio("PM schedule used in the warranty scenario",
                           ["Recommended calendar schedule", "Global optimum schedule"],
                           help="Selects which group-PM schedule the 'FRW (with PM schedule)' warranty scenario follows.")
    basis = "recommended" if basis_label.startswith("Recommended") else "best"
    if st.button("Reset all to defaults"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------------ header
st.title("Industrial Helical Gearbox — Preventive Maintenance & Warranty Optimizer")
st.caption("Two-stage helical gearbox, ~75 kW, continuous conveyor duty · Weibull reliability · renewal theory · "
           "group maintenance · warranty economics")

tabs = st.tabs(["Overview", "System & Data", "Warranty Optimization", "PM Policies", "Group PM",
                "Constraints & Maintainability", "Results Analysis", "Calculation Audit", "Methodology & References", "Export"])

# ================================================================== data editor first (feeds every tab)
with tabs[1]:
    st.subheader("Assemblies, sub-assemblies and components")
    tree = D.DEFAULT_COMPONENTS.groupby(["assembly", "sub_assembly"])["component"].apply(list)
    cols = st.columns(2)
    for n, ((a, s_), comps_) in enumerate(tree.items()):
        cols[n % 2].markdown(f"**{a}** ▸ {s_}  \n" + "  \n".join(f"&nbsp;&nbsp;&nbsp;• {x}" for x in comps_))

    st.subheader("Failure modes, causes and mechanisms")
    st.caption("Failure-mode classes follow ISO 15243 (rolling bearings) and ISO 10825 (gear teeth). "
               "Source keys refer to the bibliography in the *Methodology & References* tab.")
    st.dataframe(D.FAILURE_MODES.rename(columns={"id": "ID", "component": "Component", "failure_mode": "Failure mode",
                 "failure_cause": "Failure cause", "failure_mechanism": "Failure mechanism", "source": "Source"}),
                 hide_index=True, width="stretch")

    st.subheader("Component reliability, cost and maintainability data (editable)")
    st.caption("beta = Weibull shape; eta = Weibull scale (h) derived from the reference life: "
               "eta = L / [−ln(1−p)]^(1/beta); part / collateral in Rs; t_p planned and t_f corrective repair hours "
               "(in addition to the shared shutdown hours); critical = run-to-failure not allowed; "
               "warranted = claimable under warranty. Basis and Source columns document each row.")
    edit_cols = ["id", "component", "beta", "eta_h", "part", "collateral", "t_p", "t_f", "critical", "warranted",
                 "life_basis", "source"]
    edited = st.data_editor(D.DEFAULT_COMPONENTS[edit_cols], hide_index=True, width="stretch", key="editor",
                            disabled=["id", "component", "life_basis", "source"],
                            column_config={"beta": st.column_config.NumberColumn("β", min_value=0.3, max_value=8.0, step=0.05),
                                           "eta_h": st.column_config.NumberColumn("η (h)", min_value=500, step=1000),
                                           "life_basis": st.column_config.TextColumn("Basis of life estimate", width="large"),
                                           "source": st.column_config.TextColumn("Source")})
    comps = D.DEFAULT_COMPONENTS.copy()
    for col in edited.columns:
        if col in comps.columns and col not in ("id", "component", "life_basis", "source"):
            comps[col] = edited[col].values
    comps_json = comps.to_json(orient="split")
    res = compute(comps_json, json.dumps(g), basis)
    eng, ind = res["engine"], res["individual"]

    st.subheader("Derived cost per event (traceable)")
    der = eng.c[["id", "component", "mu", "Cp_ind", "Cp_grp", "Cf", "Cw"]].copy()
    der["mu_months"] = der.mu / eng.u
    st.dataframe(der.rename(columns={"mu": "MTTF (h)", "mu_months": "MTTF (mo)", "Cp_ind": "Cp stand-alone (Rs)",
                 "Cp_grp": "Cp in shared visit (Rs)", "Cf": "Cf failure (Rs)", "Cw": "Warranty claim cost Cw (Rs)"})
                 .round(0), hide_index=True, width="stretch")
    st.caption(f"Shared visit cost S_visit = fixed {inr(g['setup_fixed'])} + (labour + downtime) × {g['setup_hours']} h = {inr(eng.S_visit)}")

    st.subheader("Configurable Site / Commercial Baseline Parameters")
    st.caption("Plant- and market-specific values. They are planning baselines with typical industrial ranges, "
               "not universal constants; replace them with plant records and supplier quotations.")
    st.dataframe(pd.DataFrame(D.BASELINE_PARAMETERS, columns=["Key", "Parameter", "Unit", "Typical range", "Note"])
                 .assign(Current=lambda d: [str(g.get(k, "-")) for k in d.Key]),
                 hide_index=True, width="stretch")
    st.info(D.PLANNING_NOTE)

# ================================================================== OVERVIEW
with tabs[0]:
    grp = res["group"]
    best, rec = grp["best"], grp["recommended"]
    war = res["warranty"]
    wk = "FRW (with PM schedule)"
    st.subheader("Decision summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Recommended base interval T0", f"{rec['T0']:.0f} mo")
    k2.metric("Group-PM cost (recommended)", f"{inr(rec['cost_pm'])}/mo")
    k3.metric("System availability", f"{rec['avail']*100:.2f}%")
    k4.metric("Optimal warranty (FRW)", f"{war['FRW (no PM)']['best'].W:.1f} mo")
    k5.metric("Profit / unit @ optimum", inr(war["FRW (no PM)"]["best"].profit))
    st.markdown("This application selects, for each component of the gearbox, a maintenance policy "
                "(run-to-failure, age-based, time-based or grouped time-based replacement), builds a shared-shutdown "
                "calendar, and finds the warranty length that maximises manufacturer profit per unit. "
                "Every recommendation carries an engineering or economic reason.")
    st.latex(r"F(t)=1-e^{-(t/\eta)^\beta},\quad \mu=\eta\,\Gamma(1+1/\beta),\quad M(t)=F(t)+\int_0^t M(t-x)\,dF(x)")
    st.latex(r"\text{RTF: }\tfrac{C_f}{\mu}\qquad \text{Age PM: }\tfrac{C_pR(T)+C_fF(T)}{\int_0^T R}\qquad \text{Time PM: }\tfrac{C_p+C_fM(T)}{T}")
    st.latex(r"\min\; C_{sys}=\sum_i C_i(\text{policy}_i,T_i)+\frac{S_{visit}\,f_{visit}}{T_0}\ \ \text{s.t. } R_i(T_i)\ge R_{min},\ A\ge A_{min},\ \text{cost/yr}\le B")
    st.latex(r"\max_W\ \Pi(W)=P_0+\Delta P(W)-C_{prod}-\sum_i C_{w,i}M_i(W)")
    st.markdown("**Maintenance plan (recommended calendar)**")
    st.dataframe(res["plan_rec"][0][["id", "component", "policy", "interval_mo", "saving_pct", "reason"]]
                 .rename(columns={"interval_mo": "Interval (mo)", "saving_pct": "Saving vs RTF %", "reason": "Reason"}).round(1),
                 hide_index=True, width="stretch")
    st.caption("Data provenance: reliability parameters are derived from published Weibull slopes and reference lives; "
               "costs, repair times and commercial parameters are configurable baselines. See *System & Data* and "
               "*Methodology & References*.")

# ================================================================== WARRANTY
with tabs[2]:
    st.subheader("Warranty optimization")
    st.markdown("The warranty length **W** that maximises manufacturer profit per unit is sought: the customer pays a premium "
                "that rises and saturates with W, while expected claim cost rises with W. Free-replacement (FRW) and pro-rata (PRW) "
                f"schemes are compared, plus FRW when the customer follows the **{basis_label.lower()}**.")
    b = war["FRW (no PM)"]["best"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal FRW length", f"{b.W:.1f} months")
    k2.metric("Expected claim cost @W*", inr(b.wcost))
    k3.metric("Expected claims / unit", f"{b.claims:.2f}")
    k4.metric("Profit / unit @W*", inr(b.profit))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Profit per unit vs warranty length", "Expected claim cost vs W"))
    cmap = {"FRW (no PM)": BLUE, "PRW (no PM)": GREEN, "FRW (with PM schedule)": ORANGE}
    for name, v in war.items():
        cv = v["curve"]
        dash = "dash" if "PM sch" in name else "solid"
        fig.add_trace(go.Scatter(x=cv.W, y=cv.profit, name=name, line=dict(color=cmap[name], dash=dash)), 1, 1)
        fig.add_trace(go.Scatter(x=cv.W, y=cv.wcost, name=name, showlegend=False, line=dict(color=cmap[name], dash=dash)), 1, 2)
        fig.add_trace(go.Scatter(x=[v["best"].W], y=[v["best"].profit], mode="markers",
                                 marker=dict(size=11, color=cmap[name], symbol="star"), showlegend=False), 1, 1)
        bad = cv[~cv.feasible]
        if len(bad):
            fig.add_trace(go.Scatter(x=bad.W, y=bad.profit, mode="markers", marker=dict(size=4, color=RED),
                                     name="infeasible", showlegend=False), 1, 1)
    fig.update_xaxes(title_text="Warranty length W (months)")
    fig.update_yaxes(title_text="Rs", col=1)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("★ optimum · red dots = constraint violated (claim cost above the budget fraction or claims above the limit).")
    c1, c2 = st.columns(2)
    with c1:
        Wsel = st.slider("Inspect claims at W (months)", float(g["w_min"]), float(g["w_max"]), float(b.W), 0.5)
        bd = eng.warranty_breakdown(Wsel)
        bd = bd[bd.claims > 0].sort_values("cost", ascending=False)
        f2 = go.Figure(go.Bar(x=bd.id, y=bd.cost, marker_color=[ORANGE if x < 1 else BLUE for x in bd.beta],
                              hovertext=bd.component))
        f2.update_layout(height=330, title=f"Expected claim cost by component at W={Wsel:.1f} mo (orange = β<1, early life)",
                         yaxis_title="Rs per unit")
        st.plotly_chart(f2, width="stretch")
    with c2:
        st.dataframe(bd.assign(claims=bd.claims.round(4), cost=bd.cost.round(0))[["id", "component", "beta", "claims", "cost"]],
                     hide_index=True, width="stretch")
        top = bd.head(3)
        sched_note = ""
        pmw, nopm = war[wk]["best"], war["FRW (no PM)"]["best"]
        if abs(pmw.profit - nopm.profit) < 1e-6:
            sched_note = (" The selected PM schedule does not change the warranty optimum because the first planned "
                          "replacements fall after the warranty window.")
        st.markdown(f"At W = {Wsel:.1f} months the dominant claim contributors are "
                    f"{', '.join(top.id + ' (' + top.component + ')')}. "
                    f"Claim cost at W* is {b.wcost / (g['base_price'] + eng.premium(b.W)) * 100:.1f}% of the price.{sched_note}")

# ================================================================== PM POLICIES
with tabs[3]:
    st.subheader("Policy selection per component (stand-alone, each with its own shutdown)")
    show = ind[["id", "component", "beta", "ratio", "rtf_rate_pm", "age_T_mo", "age_rate_pm", "blk_T_mo", "blk_rate_pm",
                "policy", "T_mo", "saving_pct", "reason"]]
    st.dataframe(show.rename(columns={"ratio": "Cf/Cp", "rtf_rate_pm": "RTF Rs/mo", "age_T_mo": "Age T* (mo)", "age_rate_pm": "Age Rs/mo",
                 "blk_T_mo": "Time T* (mo)", "blk_rate_pm": "Time Rs/mo", "policy": "Chosen policy", "T_mo": "Chosen T (mo)",
                 "saving_pct": "Saving vs RTF %", "reason": "Reason"}).round(1), hide_index=True, width="stretch")
    pm_idx = int(ind.index[ind.policy != "Run-to-failure"][0]) if (ind.policy != "Run-to-failure").any() else 0
    cid = st.selectbox("Cost-rate curves for component", list(eng.c.index),
                       format_func=lambda i: f"{eng.c.id[i]} - {eng.c.component[i]}", index=pm_idx)
    cv = eng.curves_for(cid)
    f = go.Figure()
    for col, colr in (("Age-based", BLUE), ("Time-based", ORANGE), ("Run-to-failure", GREY)):
        f.add_trace(go.Scatter(x=cv.T_months, y=cv[col], name=col, line=dict(color=colr, dash="dot" if col == "Run-to-failure" else "solid")))
    r_ = ind.loc[cid]
    f.add_trace(go.Scatter(x=[r_.age_T_mo], y=[r_.age_rate_pm], mode="markers", marker=dict(size=11, color=BLUE, symbol="star"), name="Age optimum"))
    f.add_trace(go.Scatter(x=[r_.blk_T_mo], y=[r_.blk_rate_pm], mode="markers", marker=dict(size=11, color=ORANGE, symbol="star"), name="Time optimum"))
    if eng.c.critical[cid]:
        f.add_vline(x=eng.c.T_R_months[cid], line_dash="dash", line_color=RED, annotation_text=f"R≥{g['r_min']} limit")
    rtf_v = cv["Run-to-failure"].iloc[0]
    f.update_yaxes(range=[0, max(2.5 * rtf_v, 1.4 * min(r_.age_rate_pm, r_.blk_rate_pm))])
    f.update_layout(height=420, xaxis_title="PM interval T (months)", yaxis_title="Expected cost rate (Rs / month)",
                    title=f"{eng.c.id[cid]} {eng.c.component[cid]} (β={eng.c.beta[cid]}, η={eng.c.eta_h[cid]:,.0f} h)")
    st.plotly_chart(f, width="stretch")
    st.caption("The y-axis is zoomed around the run-to-failure level; very short intervals are far more expensive and are cut off.")
    st.markdown(f"**Decision:** {r_.policy} — {r_.reason}")

# ================================================================== GROUP PM
with tabs[4]:
    st.subheader("Component groups and PM interval of each group")
    st.markdown("A **base interval T0** is chosen; each component is serviced every **k × T0** (or run to failure). "
                "Components with the same k form a group and share one shutdown, so the visit cost S_visit is paid once per visit. "
                "The solver sweeps T0, lets every component pick its best k, then refines with a coordinate-descent search "
                "using the exact shared-visit frequency.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal base interval T0*", f"{best['T0']:.2f} mo")
    k2.metric("Group-PM cost", f"{inr(best['cost_pm'])}/month")
    k3.metric("vs independent time-based PM", f"{grp['saving_vs_indep_block_pct']:.1f}% lower")
    k4.metric("System availability", f"{best['avail']*100:.2f}%")
    sw = grp["sweep"]
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Scatter(x=sw.T0_months, y=sw.cost_per_month, name="Total cost (Rs/month)", line=dict(color=BLUE)), secondary_y=False)
    f.add_trace(go.Scatter(x=sw.T0_months, y=sw.availability * 100, name="Availability (%)", line=dict(color=GREEN, dash="dot")), secondary_y=True)
    bad = sw[~sw.feasible]
    if len(bad):
        f.add_trace(go.Scatter(x=bad.T0_months, y=bad.cost_per_month, mode="markers", marker=dict(color=RED, size=4),
                               name="constraint violated"), secondary_y=False)
    f.add_trace(go.Scatter(x=[best["T0"]], y=[best["cost_pm"]], mode="markers", marker=dict(size=13, color=ORANGE, symbol="star"),
                           name="optimum"), secondary_y=False)
    f.update_yaxes(title_text="Rs / month", secondary_y=False)
    f.update_yaxes(title_text="Availability %", secondary_y=True)
    f.update_xaxes(title_text="Base interval T0 (months)")
    f.update_layout(height=400, legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(f, width="stretch")
    if not grp["feasible_found"]:
        st.warning("No base interval satisfies every constraint (availability and annual budget); the least-cost interval is shown.")
    st.markdown("**Practical (round) base intervals**")
    pr = grp["practical"].copy()
    pr["cost_per_month"] = pr.cost_per_month.round(0)
    pr["availability"] = (pr.availability * 100).round(2)
    st.dataframe(pr.rename(columns={"T0_months": "T0 (mo)", "cost_per_month": "Rs/month", "availability": "Avail %",
                 "visits_per_year": "Visits/yr", "penalty_pct": "% above optimum"}).round(2), hide_index=True, width="stretch")
    mode = st.radio("Show the schedule for", ["Recommended practical schedule", "Continuous optimum", "Choose my own T0"], horizontal=True)
    if mode == "Choose my own T0":
        T0u = st.slider("Base interval T0 (months)", float(g["t_min_months"]), 24.0, 3.0, 0.5)
        ev = eng.eval_group(T0u)
        plan, groups = M.group_plan(eng, ind, ev)
    elif mode == "Continuous optimum":
        ev, (plan, groups) = best, res["plan_best"]
    else:
        ev, (plan, groups) = rec, res["plan_rec"]
    st.markdown(f"**Schedule at T0 = {ev['T0']:.2f} months** — cost {inr(ev['cost_pm'])}/month ({inr(ev['cost_py'])}/yr), "
                f"availability {ev['avail']*100:.2f}%, {ev['visits_per_year']:.2f} shutdowns/year, feasible: {'✅' if ev['feasible'] else '❌'}")
    st.dataframe(groups.rename(columns={"interval_months": "Interval (mo)", "events_per_year": "Events/yr", "components": "Components",
                 "ids": "IDs", "cost_per_event": "Cost/event (Rs)", "downtime_h_per_event": "Downtime/event (h)"}).round(1),
                 hide_index=True, width="stretch")
    st.caption("Downtime per event assumes a single crew working sequentially: planned repair hours of all components in the "
               "visit are summed (conservative), plus the shared shutdown hours.")
    st.markdown("**Reason for each maintenance requirement**")
    st.dataframe(plan[["id", "component", "policy", "interval_mo", "saving_pct", "reason"]]
                 .rename(columns={"interval_mo": "Interval (mo)", "saving_pct": "Saving vs RTF %", "reason": "Reason"}).round(1),
                 hide_index=True, width="stretch")
    horizon = 120
    ft = go.Figure()
    for j, r in groups.iterrows():
        xs = np.arange(r.interval_months, horizon + 1e-9, r.interval_months)
        ft.add_trace(go.Scatter(x=xs, y=[r.group] * len(xs), mode="markers", marker=dict(size=11, symbol="diamond"),
                                name=f"{r.group}: {r.ids}", hovertext=r.components))
    ft.update_layout(height=60 * max(len(groups), 1) + 180, xaxis_title="Months in service", title="Maintenance calendar (first 10 years)")
    st.plotly_chart(ft, width="stretch")
    st.markdown("**Comparison of strategies (Rs / month)**")
    cmp_ = pd.DataFrame({"Strategy": ["All run-to-failure (reference)", "Independent best policy (age/time/RTF)", "Independent time-based PM",
                                      "Group PM (optimum)", f"Group PM (practical T0={rec['T0']:.0f})"],
                         "Rs/month": [grp["all_rtf_pm"], grp["independent_pm"], grp["independent_block_pm"], best["cost_pm"], rec["cost_pm"]]})
    fb = go.Figure(go.Bar(x=cmp_.Strategy, y=cmp_["Rs/month"], marker_color=[GREY, BLUE, BLUE, ORANGE, ORANGE],
                          text=cmp_["Rs/month"].round(0), textposition="outside"))
    fb.update_layout(height=380, yaxis_title="Rs / month", yaxis_range=[0, cmp_["Rs/month"].max() * 1.15])
    st.plotly_chart(fb, width="stretch")
    st.caption("Like-for-like: group PM is time-based, so compare it with independent time-based PM. Independent age-based PM "
               "is cheaper per part but needs every part's age tracked and its own shutdown. Safety-critical parts cannot run to "
               "failure, so 'all run-to-failure' is a reference level only and can be below any feasible plan.")

# ================================================================== CONSTRAINTS
with tabs[5]:
    st.subheader("Objective, constraints and maintainability")
    ev = rec
    _pl = res["plan_rec"][0]
    _R = [float(eng.R_at(i, r_.interval_mo * eng.u)) for i, r_ in _pl.iterrows()
          if eng.c.critical[i] and eng.c.beta[i] > 1.05 and r_.k > 0]
    crit_minR = min(_R) if _R else 1.0
    int_min = np.nanmin(_pl.interval_mo) if _pl.interval_mo.notna().any() else np.inf
    wb = war["FRW (no PM)"]["best"]
    tab_ = [
        ("Objective", "Minimise total maintenance cost rate (Rs/month)", f"{inr(ev['cost_pm'])}/month", "-"),
        ("Reliability", f"R(T) ≥ {g['r_min']} for safety-critical parts", f"min R(T) = {crit_minR:.3f}",
         "✅" if crit_minR >= g["r_min"] - 1e-9 else "❌"),
        ("Availability", f"A ≥ {g['a_min']}", f"{ev['avail']:.4f}", "✅" if ev["avail"] >= g["a_min"] else "❌"),
        ("Budget", f"Cost ≤ {inr(g['budget_per_year'])}/yr", f"{inr(ev['cost_py'])}/yr", "✅" if ev["cost_py"] <= g["budget_per_year"] else "❌"),
        ("Minimum interval", f"T ≥ {g['t_min_months']} months", f"min interval {int_min:.1f} mo", "✅" if int_min >= g["t_min_months"] else "❌"),
        ("Warranty cost", f"≤ {g['warranty_budget_frac']*100:.0f}% of price", inr(wb.wcost), "✅" if wb.feasible else "❌"),
        ("Warranty claims", f"≤ {g['max_claims']} per unit", f"{wb.claims:.2f}", "✅" if wb.claims <= g["max_claims"] else "❌"),
    ]
    st.dataframe(pd.DataFrame(tab_, columns=["Item", "Requirement", "Achieved", "OK"]), hide_index=True, width="stretch")
    st.markdown("**Maintainability view** — downtime caused by each component per 1000 operating hours in the recommended plan")
    pl = res["plan_rec"][0].sort_values("downtime_h_per_1000h", ascending=False)
    fm = go.Figure(go.Bar(x=pl.id, y=pl.downtime_h_per_1000h, marker_color=BLUE, hovertext=pl.component))
    fm.update_layout(height=320, yaxis_title="downtime h / 1000 h")
    st.plotly_chart(fm, width="stretch")
    st.markdown("**What do the constraints cost?** (re-optimising for each setting)")
    rs, as_ = cached_sweeps(comps_json, json.dumps(g))
    c1, c2 = st.columns(2)
    with c1:
        fr = go.Figure(go.Scatter(x=rs.r_min, y=rs.cost_pm, mode="lines+markers", line=dict(color=RED)))
        fr.update_layout(height=320, title="Cost vs required R_min", xaxis_title="R_min (critical parts)", yaxis_title="Rs/month")
        st.plotly_chart(fr, width="stretch")
    with c2:
        fa = go.Figure(go.Scatter(x=as_.a_min, y=as_.cost_pm, mode="lines+markers", line=dict(color=GREEN),
                                  marker=dict(color=[GREEN if f_ else RED for f_ in as_.feasible])))
        fa.update_layout(height=320, title="Cost vs required availability (red = infeasible)", xaxis_title="A_min", yaxis_title="Rs/month")
        st.plotly_chart(fa, width="stretch")

# ================================================================== ANALYSIS
with tabs[6]:
    st.subheader("Results analysis")
    delta = st.slider("Sensitivity perturbation (±)", 0.05, 0.50, 0.20, 0.05)
    base, sens = cached_sens(comps_json, json.dumps(g), delta)
    st.markdown("**1. One-at-a-time sensitivity**")
    s = sens.copy()
    s["swing_cost"] = (s.cost_hi - s.cost_lo).abs()
    s = s.sort_values("swing_cost")
    ft = go.Figure()
    ft.add_trace(go.Bar(y=s.parameter, x=s.cost_lo - base["grouped_cost_pm"], orientation="h", name=f"-{delta*100:.0f}%", marker_color=BLUE))
    ft.add_trace(go.Bar(y=s.parameter, x=s.cost_hi - base["grouped_cost_pm"], orientation="h", name=f"+{delta*100:.0f}%", marker_color=ORANGE))
    ft.update_layout(barmode="overlay", height=430, xaxis_title=f"Change in group-PM cost (Rs/month) from base {base['grouped_cost_pm']:,.0f}")
    st.plotly_chart(ft, width="stretch")
    s2 = sens.copy()
    s2["swing_W"] = (s2.W_hi - s2.W_lo).abs()
    s2 = s2.sort_values("swing_W")
    fw = go.Figure()
    fw.add_trace(go.Bar(y=s2.parameter, x=s2.W_lo - base["W_opt_PM"], orientation="h", name=f"-{delta*100:.0f}%", marker_color=BLUE))
    fw.add_trace(go.Bar(y=s2.parameter, x=s2.W_hi - base["W_opt_PM"], orientation="h", name=f"+{delta*100:.0f}%", marker_color=ORANGE))
    fw.update_layout(barmode="overlay", height=430, xaxis_title=f"Change in optimal warranty length (months) from base {base['W_opt_PM']:.1f}")
    st.plotly_chart(fw, width="stretch")
    top = s.iloc[::-1].head(3).parameter.tolist()
    topw = s2.iloc[::-1].head(2).parameter.tolist()
    st.markdown("**2. Monte-Carlo validation** — simulated vs analytical formulas")
    mc_idx = int(eng.c.mu.idxmin())
    cid2 = st.selectbox("Component", list(eng.c.index), format_func=lambda i: f"{eng.c.id[i]} - {eng.c.component[i]}", index=mc_idx, key="mc_c")
    Tmc = st.slider("Interval T (months)", 4.0, 120.0, 40.0, 2.0)
    v = eng.mc_validate(cid2, Tmc, n=60000)
    st.dataframe(pd.DataFrame({"Quantity": ["Renewal function M(T)", "Time-based cost (Rs/mo)", "Age-based cost (Rs/mo)"],
                               "Analytical": [v["M_analytic"], v["block_analytic"], v["age_analytic"]],
                               "Monte-Carlo (60 000 runs)": [v["M_mc"], v["block_mc"], v["age_mc"]]}).round(4),
                 hide_index=True, width="stretch")
    st.markdown("**3. Key findings (generated from the current inputs)**")
    sav_ind = grp["saving_vs_indep_block_pct"]
    crit = ind[ind.crit_wear]
    n_pm = int((res["plan_rec"][0].k > 0).sum())
    n_rtf = len(ind) - n_pm
    pr_ = grp["practical"]
    near = pr_.loc[pr_.penalty_pct < 3, "T0_months"]
    near_txt = f"{near.min():.0f} to {near.max():.0f}" if len(near) else "no practical value"
    r_near = rs.loc[rs.r_min.sub(g["r_min"]).abs().idxmin(), "cost_pm"]
    st.markdown(f"""
- **Policy mix:** {n_pm} of {len(ind)} components receive planned replacement in the recommended plan, {n_rtf} run to failure, each with a stated reason.
- **Grouping:** sharing the shutdown ({inr(eng.S_visit)} per visit) changes cost by **{sav_ind:.1f}%** relative to independent time-based PM.
- **Interval flexibility:** base intervals of {near_txt} months are within 3% of the optimum.
- **Reliability constraint:** R ≥ {g['r_min']} on {len(crit)} safety-critical parts gives cost {inr(r_near)}/month, {(r_near / rs.cost_pm.min() - 1)*100:.1f}% above the least-restrictive setting in the sweep.
- **Most influential inputs for cost:** {', '.join(top)}. For warranty length: {', '.join(topw)}.
""")
    st.warning("Limitations: independent components; perfect (as-good-as-new) replacement; constant costs; no condition monitoring; "
               "no discounting; group schedule solved by a heuristic (coordinate descent); two-parameter Weibull models; "
               "cost and repair-time baselines to be replaced with plant records.")

# ================================================================== CALCULATION AUDIT
with tabs[7]:
    st.subheader("Calculation Audit — step-by-step derivation")
    cid3 = st.selectbox("Component", list(eng.c.index), format_func=lambda i: f"{eng.c.id[i]} - {eng.c.component[i]}", index=pm_idx, key="ex_c")
    T_def = float(min(max(round(float(ind.blk_T_mo[cid3])), 2.0), 150.0))
    T_ex = st.slider("Evaluate PM at T (months)", 2.0, 150.0, T_def, 1.0)
    for title, body in eng.explain_component(cid3, T_ex):
        st.markdown(f"**{title}**")
        st.code(body, language=None)
    st.markdown("**Renewal function** — solve M(t)=F(t)+∫M(t−x)dF(x) on a grid with the trapezoid rule:")
    st.latex(r"M_n=\frac{F_n+\tfrac12 M_{n-1}\Delta F_1+\sum_{k=2}^{n}\tfrac12\,(M_{n-k+1}+M_{n-k})\,\Delta F_k}{1-\tfrac12\Delta F_1}")
    cvm = go.Figure()
    tt = eng.t / eng.u
    mask = tt <= 150
    cvm.add_trace(go.Scatter(x=tt[mask], y=eng.M[cid3][mask], name="M(t) renewal function", line=dict(color=BLUE)))
    cvm.add_trace(go.Scatter(x=tt[mask], y=eng.F[cid3][mask], name="F(t) first-failure CDF", line=dict(color=ORANGE, dash="dash")))
    cvm.update_layout(height=340, xaxis_title="months", yaxis_title="expected failures / probability")
    st.plotly_chart(cvm, width="stretch")
    st.caption("M(t) exceeds F(t) once repeat failures become possible; this is why block (time-based) PM needs M(T), not F(T).")

# ================================================================== METHODOLOGY & REFERENCES
with tabs[8]:
    st.subheader("Methodology")
    st.markdown("**Reliability.** Two-parameter Weibull model per component; lives quoted at a failure probability p are converted to the scale parameter by")
    st.latex(r"\eta=\frac{L_p}{[-\ln(1-p)]^{1/\beta}},\qquad L_{10}=\eta\,(-\ln 0.90)^{1/\beta}\ \ (\text{ISO 281, }p=0.10)")
    st.markdown("**Maintenance policies and cost rates (per operating hour).**")
    st.latex(r"\text{RTF: }\frac{C_f}{\mu}\quad\text{Age: }\frac{C_pR(T)+C_fF(T)}{\int_0^T R(x)dx}\quad\text{Time: }\frac{C_p+C_fM(T)}{T}")
    st.latex(r"C_p=\text{part}+(L+D)(t_p+h_s)+S_f,\qquad C_f=\text{part}+\text{collateral}+(L+D)(t_f+h_s)+S_f")
    st.latex(r"\text{Group PM: } C(T_0)=\sum_i \min_{k_i}\frac{C_{p,i}'+C_{f,i}M_i(k_iT_0)}{k_iT_0}+\frac{S_{visit}\,f(\{k_i\})}{T_0},\qquad A=\frac{1}{1+\sum_i \text{downtime rate}_i}")
    st.latex(r"\text{Warranty: FRW } E[C_w]=\sum_i C_{w,i}M_i(W);\ \ \text{PRW } E[C_w]=\sum_i C_{w,i}\tfrac1W\!\int_0^W\! M_i;\ \ \Pi(W)=P_0+\Delta P(W)-C_{prod}-E[C_w],\ \Delta P(W)=\Delta P_{cap}(1-e^{-W/\tau})")
    st.subheader("Modelling assumptions")
    st.markdown(f"""
1. **Sequential single crew (conservative).** Components that share a shutdown are replaced one after another by one crew, so planned repair hours t_p are **summed** within a maintenance group, in addition to the shared shutdown hours.
2. **Shared visit.** Fixed visit cost and shutdown hours are paid once per visit; each component keeps its own part cost and repair time.
3. **As-good-as-new renewal.** Planned replacement and corrective repair both restore a component to new condition; components fail independently.
4. **Two-parameter Weibull.** No failure-free interval is modelled; for gears and shafts this is conservative relative to the three-parameter form recommended in the literature.
5. **Planning horizon.** Renewal and cost curves are evaluated on a {eng.HMAX_MONTHS/12:.0f}-year grid ({eng.HMAX_MONTHS:.0f} months). When the cost-optimal PM interval of an unconstrained wear-out component lies beyond it, the component runs to failure and the reason states this explicitly.
6. **Safety-critical parts.** Wear-out parts flagged critical must be replaced before R(T) falls below R_min; run-to-failure is not allowed.
7. **Group intervals.** Allowed multiples k of the base interval are those of {M.KSET} not exceeding k_max; the exact shared-visit frequency is computed from the least common multiple of the selected k values.
8. **Warranty.** Only components flagged *warranted* generate claims; lubricant is excluded (routine duty). The PM schedule used in the PM warranty scenario is selectable in the sidebar (recommended calendar schedule or global optimum).
9. **Currency and time.** Costs in INR (Rs); lives in operating hours; {g['hours_per_month']:.0f} operating hours per month.
""")
    st.subheader("Data provenance")
    st.info(D.PLANNING_NOTE)
    st.subheader("Engineering references")
    refs = pd.DataFrame([dict(Key=k, Reference=v["citation"], Supports=v["supports"]) for k, v in D.REFERENCES.items()])
    st.dataframe(refs, hide_index=True, width="stretch",
                 column_config={"Reference": st.column_config.TextColumn(width="large"),
                                "Supports": st.column_config.TextColumn(width="large")})

# ================================================================== EXPORT
with tabs[9]:
    st.subheader("Export results")
    tables = M.export_tables(res, g, {"Failure modes": D.FAILURE_MODES, "References": pd.DataFrame(
        [dict(key=k, **v) for k, v in D.REFERENCES.items()])})

    def make_xlsx(tbls):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for name, df in tbls.items():
                df.to_excel(xw, sheet_name=name, index=False)
        return buf.getvalue()

    def make_csv_zip(tbls):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, df in tbls.items():
                zf.writestr(name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv", df.to_csv(index=False))
        return buf.getvalue()

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ All tables (Excel .xlsx)", make_xlsx(tables), "gearbox_pm_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇️ All tables (CSV .zip)", make_csv_zip(tables), "gearbox_pm_results_csv.zip", "application/zip")
    c3.download_button("⬇️ Inputs (JSON)", json.dumps({"parameters": g, "components": json.loads(comps_json)}, indent=1),
                       "gearbox_inputs.json", "application/json")
    pick = st.selectbox("Single table (CSV)", list(tables.keys()))
    st.download_button("⬇️ Download selected table (CSV)", tables[pick].to_csv(index=False), f"{pick.replace(' ', '_')}.csv", "text/csv")
    st.dataframe(tables[pick], hide_index=True, width="stretch")
