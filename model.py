"""
model.py  --  Preventive-Maintenance and Warranty optimisation engine
====================================================================
All internal time is in OPERATING HOURS; the public API accepts/returns MONTHS
where stated (months x hours_per_month = hours).

Mathematics implemented (every formula is also shown step-by-step by
`Engine.explain_component`):

  Weibull        F(t)=1-exp[-(t/eta)^beta]   R=1-F   mu=eta*Gamma(1+1/beta)
  Renewal fn     M(t)=F(t)+int_0^t M(t-x) dF(x)        (trapezoid scheme, O(dt^2))

  Run-to-failure (RTF)   C = Cf / mu
  Age-based PM           C(T) = [Cp R(T) + Cf F(T)] / int_0^T R(x)dx
  Time(block)-based PM   C(T) = [Cp + Cf M(T)] / T
  Group PM               min_{T0} sum_i min_{k_i} [Cp_i' + Cf_i M_i(k_i T0)]/(k_i T0)  +  S_visit/T0
  Warranty (FRW)         E[Cw(W)] = sum_i Cw_i M_i(W)
  Warranty (PRW)         E[Cw(W)] = sum_i Cw_i (1/W) int_0^W M_i(x)dx
  Warranty optimisation  max_W  Profit(W) = P0 + Premium(W) - Cprod - E[Cw(W)]

Modelling assumptions worth stating explicitly
  * Sequential crew: components sharing a shutdown are replaced one after another by a single
    crew, so planned repair hours t_p ADD UP within a maintenance group (conservative).
  * Group PM shares the visit set-up cost/hours; each component keeps its own part and repair cost.
  * Planned replacement renews a component "as good as new"; corrective repair likewise.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma


# ----------------------------------------------------------------- helpers
def weibull_F(t, beta, eta):
    t = np.asarray(t, float)
    return 1.0 - np.exp(-(t / eta) ** beta)

def weibull_mean(beta, eta):
    return eta * _gamma(1.0 + 1.0 / beta)

def renewal_function(beta, eta, dt, n):
    """Renewal function M(t_k), t_k=k*dt, k=0..n, trapezoid discretisation of
    M(t)=F(t)+int_0^t M(t-x)dF(x)  (Xie-type scheme, second-order accurate)."""
    t = np.arange(n + 1) * dt
    F = weibull_F(t, beta, eta)
    dF = np.diff(F, prepend=0.0)            # dF[k]=F[k]-F[k-1], dF[0]=0
    M = np.zeros(n + 1)
    for m in range(1, n + 1):
        # sum_{k=2}^{m} 0.5*(M[m-k+1]+M[m-k])*dF[k]
        if m >= 2:
            s = 0.5 * np.dot(M[m - 1:0:-1] + M[m - 2::-1][:m - 1], dF[2:m + 1])
        else:
            s = 0.0
        M[m] = (F[m] + 0.5 * M[m - 1] * dF[1] + s) / (1.0 - 0.5 * dF[1])
    return M


KSET = (1, 2, 3, 4, 6, 8, 12, 16, 24)     # candidate multiples of the base interval T0 (capped by k_max)


@lru_cache(maxsize=None)
def visit_fraction(ks: tuple) -> float:
    """Fraction of base intervals n*T0 at which at least one component (with
    multiple k) is due -> exact shared-visit frequency = fraction/T0."""
    if not ks:
        return 0.0
    P = math.lcm(*ks)
    n = np.arange(1, P + 1)
    due = np.zeros(P, bool)
    for k in ks:
        due |= (n % k == 0)
    return float(due.mean())


# ----------------------------------------------------------------- engine
class Engine:
    HMAX_MONTHS = 180.0                    # time-grid horizon (15 years)

    def __init__(self, comps: pd.DataFrame, g: dict):
        self.g = dict(g)
        self.c = comps.reset_index(drop=True).copy()
        u = self.u = float(g["hours_per_month"])
        self.dt = u / 8.0
        kmax = int(g.get("k_max", max(KSET)))
        self.kset = tuple(k for k in KSET if k <= kmax) or (1,)
        self.n = int(round(self.HMAX_MONTHS * u / self.dt))
        self.t = np.arange(self.n + 1) * self.dt
        self._derive()
        self._build_curves()

    # ---- cost derivation (every term is traceable)
    def _derive(self):
        c, g = self.c, self.g
        L, D, Sf, hs = g["labour_rate"], g["downtime_cost"], g["setup_fixed"], g["setup_hours"]
        c["mu"] = [weibull_mean(b, e) for b, e in zip(c.beta, c.eta_h)]
        c["Cp_ind"] = c.part + (L + D) * (c.t_p + hs) + Sf       # stand-alone planned replacement
        c["Cp_grp"] = c.part + (L + D) * c.t_p                   # inside a shared visit (setup excluded)
        c["Cf"] = c.part + c.collateral + (L + D) * (c.t_f + hs) + Sf
        warr_dt = D if g["warranty_covers_downtime"] else 0.0
        c["Cw"] = c.part + c.collateral + (L + warr_dt) * (c.t_f + hs) + Sf
        c["Cpw_grp"] = c.part + L * c.t_p                        # PM paid by manufacturer (no downtime)
        c["Dp_ind"] = c.t_p + hs
        c["Dp_grp"] = c.t_p
        c["Df"] = c.t_f + hs
        self.S_visit = Sf + (L + D) * hs
        self.D_visit = hs
        # reliability-limited maximum interval for safety-critical parts
        rmin = g["r_min"]
        tr = c.eta_h * (-math.log(rmin)) ** (1.0 / c.beta)
        hmax_h = self.n * self.dt
        c["Tmax_h"] = np.where(c.critical, np.minimum(tr, hmax_h), hmax_h)
        c["T_R_months"] = tr / self.u

    def _build_curves(self):
        self.F, self.R, self.M, self.I = [], [], [], []
        for _, r in self.c.iterrows():
            F = weibull_F(self.t, r.beta, r.eta_h)
            R = 1 - F
            I = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * self.dt)])
            self.F.append(F); self.R.append(R); self.I.append(I)
            self.M.append(renewal_function(r.beta, r.eta_h, self.dt, self.n))

    # ---- point evaluators (T in hours)
    def M_at(self, i, T):  return np.interp(T, self.t, self.M[i])
    def F_at(self, i, T):  return np.interp(T, self.t, self.F[i])
    def R_at(self, i, T):  return np.interp(T, self.t, self.R[i])
    def I_at(self, i, T):  return np.interp(T, self.t, self.I[i])

    def rate(self, kind, i, T, Cp, Cf):
        """Generic cost (or downtime-hour) rate per operating hour."""
        if kind == "rtf":   return Cf / self.c.mu[i]
        if kind == "age":   return (Cp * self.R_at(i, T) + Cf * self.F_at(i, T)) / self.I_at(i, T)
        if kind == "block": return (Cp + Cf * self.M_at(i, T)) / T
        raise ValueError(kind)

    # ================================================== 1. individual policies
    def individual_policies(self):
        g, c, u = self.g, self.c, self.u
        tmin = g["t_min_months"] * u
        rows = []
        for i, r in c.iterrows():
            tmax = r.Tmax_h
            idx = np.arange(int(math.ceil(tmin / self.dt)), int(tmax / self.dt) + 1)
            T = self.t[idx]
            age = (r.Cp_ind * self.R[i][idx] + r.Cf * self.F[i][idx]) / self.I[i][idx]
            blk = (r.Cp_ind + r.Cf * self.M[i][idx]) / T
            ja, jb = int(np.argmin(age)), int(np.argmin(blk))
            rtf = r.Cf / r.mu
            # an optimum sitting on the horizon boundary (unconstrained part) => no finite optimum
            at_edge = lambda j: (j >= len(idx) - 1) and not r.critical
            age_ok, blk_ok = not at_edge(ja), not at_edge(jb)
            beyond_horizon = (not age_ok) or (not blk_ok)
            critical_wear = bool(r.critical and r.beta > 1.05)
            cand = {}
            if not critical_wear: cand["Run-to-failure"] = (rtf, np.nan)
            if age_ok: cand["Age-based PM"] = (age[ja], T[ja])
            if blk_ok: cand["Time-based PM"] = (blk[jb], T[jb])
            best = min(cand, key=lambda k: cand[k][0])
            best_rate, best_T = cand[best]
            saving = (rtf - best_rate) / rtf
            # ---- engineering justification for the selected policy
            ratio = r.Cf / r.Cp_ind
            premium_pct = 0.0
            if best != "Run-to-failure" and not critical_wear and saving < g["pm_saving_threshold"]:
                reason = (f"Best PM saves only {saving*100:.1f}% (< {g['pm_saving_threshold']*100:.0f}% threshold; "
                          f"Cf/Cp={ratio:.1f}); a planned stop costs almost as much as a failure -> run to failure")
                best, best_rate, best_T, saving = "Run-to-failure", rtf, np.nan, 0.0
            elif best == "Run-to-failure":
                if r.beta < 0.95:
                    reason = (f"beta={r.beta:.2f}<1: decreasing hazard (infant mortality) -> a planned replacement "
                              f"resets to the riskiest state; control by commissioning QA and cover by warranty")
                elif r.beta <= 1.05:
                    reason = (f"beta={r.beta:.2f}~1: constant hazard (random failures) -> replacing a working part "
                              f"gives no benefit; run to failure")
                elif beyond_horizon:
                    reason = (f"Wear-out (beta={r.beta:.2f}) but the cost-optimal PM interval exceeds the "
                              f"{self.HMAX_MONTHS/12:.0f}-year planning horizon ({self.HMAX_MONTHS:.0f} mo); "
                              f"within the horizon no PM interval beats run-to-failure (Cf/Cp={ratio:.1f})")
                else:
                    reason = (f"Wear-out (beta={r.beta:.2f}) but Cf/Cp={ratio:.1f} is too low: "
                              f"no interval beats run-to-failure")
            elif critical_wear:
                premium_pct = -saving * 100
                reason = (f"Safety/consequence-critical wear-out part (beta={r.beta:.1f}): run-to-failure not allowed; "
                          f"replace before R<{g['r_min']:.2f} (T<={r.T_R_months:.0f} mo). Constraint costs "
                          f"{premium_pct:+.1f}% vs unconstrained RTF (price of safety)")
            else:
                reason = (f"Wear-out beta={r.beta:.1f}>1 and Cf/Cp={ratio:.1f}: {best} at "
                          f"{best_T/u:.1f} mo cuts cost {saving*100:.1f}% vs RTF")
            rows.append(dict(
                id=r.id, component=r.component, beta=r.beta, eta_h=r.eta_h, mu_h=r.mu,
                Cp=r.Cp_ind, Cf=r.Cf, ratio=ratio,
                rtf_rate_pm=rtf * u,                      # Rs/month
                age_T_mo=T[ja] / u, age_rate_pm=age[ja] * u,
                blk_T_mo=T[jb] / u, blk_rate_pm=blk[jb] * u,
                T_R_mo=r.T_R_months if r.critical else np.nan,
                policy=best, T_mo=best_T / u if best_T == best_T else np.nan,
                rate_pm=best_rate * u, saving_pct=saving * 100, reason=reason,
                crit_wear=critical_wear, _age_ok=age_ok, _blk_ok=blk_ok))
        return pd.DataFrame(rows)

    def curves_for(self, i, n_pts=500, t_max_months=None):
        """Cost-rate-vs-T curves (Rs/month) for plotting one component."""
        r, u = self.c.loc[i], self.u
        tmin = self.g["t_min_months"] * u
        tmax = min(r.Tmax_h, (t_max_months or 150.0) * u)
        idx = np.unique(np.linspace(math.ceil(tmin / self.dt), int(tmax / self.dt), n_pts).astype(int))
        T = self.t[idx]
        return pd.DataFrame({
            "T_months": T / u,
            "Age-based": (r.Cp_ind * self.R[i][idx] + r.Cf * self.F[i][idx]) / self.I[i][idx] * u,
            "Time-based": (r.Cp_ind + r.Cf * self.M[i][idx]) / T * u,
            "Run-to-failure": np.full(len(idx), r.Cf / r.mu * u)})

    # ================================================== 2. group PM
    def eval_group(self, T0_months):
        """Cost, downtime and schedule at one base interval T0 (months).
        Each component picks its multiple k_i (or run-to-failure); a local
        search then minimises the EXACT total incl. shared-visit frequency."""
        u, g, c = self.u, self.g, self.c
        T0 = T0_months * u
        ks = np.array(self.kset)
        Tk = ks * T0
        opts = []
        for i, r in c.iterrows():
            ok = (Tk <= r.Tmax_h) & (Tk <= self.n * self.dt)
            cost = np.full(len(ks), np.inf)
            cost[ok] = (r.Cp_grp + r.Cf * np.interp(Tk[ok], self.t, self.M[i])) / Tk[ok]
            c0 = np.inf if (r.critical and r.beta > 1.05) else r.Cf / r.mu
            opts.append(np.concatenate([[c0], cost]))
        ch = [int(np.argmin(o)) for o in opts]

        def total(ch):
            base = sum(opts[i][j] for i, j in enumerate(ch))
            kk = tuple(sorted({self.kset[j - 1] for j in ch if j > 0}))
            return base + self.S_visit * visit_fraction(kk) / T0, kk
        best, _ = total(ch)
        improved = True
        while improved and np.isfinite(best):
            improved = False
            for i in range(len(ch)):
                for j in range(len(opts[i])):
                    if j == ch[i] or not np.isfinite(opts[i][j]):
                        continue
                    trial = ch.copy(); trial[i] = j
                    val, _ = total(trial)
                    if val < best - 1e-9:
                        best, ch, improved = val, trial, True
        tot_cost, kk = total(ch)
        f = visit_fraction(kk)
        rows, tot_dt = [], 0.0
        for i, r in c.iterrows():
            j = ch[i]
            if j == 0:
                dt_rate = r.Df / r.mu; k = 0; Tm = np.nan; rate = r.Cf / r.mu
            else:
                k = self.kset[j - 1]; Th = k * T0
                dt_rate = (r.Dp_grp + r.Df * self.M_at(i, Th)) / Th; Tm = Th / u; rate = opts[i][j]
            rows.append(dict(id=r.id, component=r.component, k=k, T_mo=Tm, rate=rate, dt=dt_rate))
            tot_dt += dt_rate
        tot_dt += self.D_visit * f / T0
        avail = 1.0 / (1.0 + tot_dt)
        per_year = tot_cost * u * 12
        feasible = bool((avail >= g["a_min"]) and (per_year <= g["budget_per_year"]) and np.isfinite(tot_cost))
        return dict(T0=T0_months, cost_pm=tot_cost * u, cost_py=per_year, avail=avail, feasible=feasible,
                    visits_per_year=f / T0 * u * 12, visit_fraction=f, ks=kk,
                    table=pd.DataFrame(rows), S_visit_pm=self.S_visit * f / T0 * u)

    def group_sweep(self, lo=None, hi=36.0, step=0.25):
        lo = lo or self.g["t_min_months"]
        grid = np.arange(lo, hi + 1e-9, step)
        res = [self.eval_group(t) for t in grid]
        df = pd.DataFrame({"T0_months": grid,
                           "cost_per_month": [r["cost_pm"] for r in res],
                           "availability": [r["avail"] for r in res],
                           "cost_per_year": [r["cost_py"] for r in res],
                           "feasible": [r["feasible"] for r in res]})
        feas = df[df.feasible]
        pool = feas if len(feas) else df
        j = int(pool.cost_per_month.idxmin())
        return df, res[j], bool(len(feas))

    def group_summary(self, ind: pd.DataFrame, T0_practical=(1, 2, 3, 4, 6, 9, 12)):
        sweep, best, feas = self.group_sweep()
        rtf_all = ind.rtf_rate_pm.sum()
        indep = ind.rate_pm.sum()                       # each part its own best policy (age/block/RTF)
        # like-for-like with group PM: calendar (block) policy only, every part with its OWN shutdown
        indep_block = float(np.where(ind.crit_wear, ind.blk_rate_pm,
                                     np.minimum(ind.rtf_rate_pm, ind.blk_rate_pm)).sum())
        prac, evals = [], {}
        for t in T0_practical:
            r = self.eval_group(float(t)); evals[t] = r
            prac.append(dict(T0_months=t, cost_per_month=r["cost_pm"], availability=r["avail"],
                             visits_per_year=r["visits_per_year"], feasible=r["feasible"],
                             penalty_pct=(r["cost_pm"] / best["cost_pm"] - 1) * 100))
        prac = pd.DataFrame(prac)
        pool = prac[prac.feasible] if prac.feasible.any() else prac
        rec_T0 = float(pool.loc[pool.cost_per_month.idxmin(), "T0_months"])
        return dict(sweep=sweep, best=best, recommended=evals[rec_T0] if rec_T0 in evals else best,
                    feasible_found=feas, all_rtf_pm=rtf_all, independent_pm=indep, independent_block_pm=indep_block,
                    grouped_pm=best["cost_pm"], saving_vs_indep_pct=(1 - best["cost_pm"] / indep) * 100,
                    saving_vs_indep_block_pct=(1 - best["cost_pm"] / indep_block) * 100,
                    practical=prac)

    # ================================================== 3. warranty
    def _failures(self, i, W_h, T_h=None):
        if T_h is None or not np.isfinite(T_h) or T_h <= 0:
            return float(self.M_at(i, W_h))
        n = math.floor(W_h / T_h + 1e-12)
        return n * float(self.M_at(i, T_h)) + float(self.M_at(i, W_h - n * T_h))

    def warranty_point(self, W_months, scheme="FRW", pm_T_months=None):
        """Expected claims and manufacturer cost per unit for warranty length W."""
        u, g, c = self.u, self.g, self.c
        W = W_months * u
        cost, claims, pm_cost = 0.0, 0.0, 0.0
        for i, r in c.iterrows():
            if not r.warranted:
                continue
            Tm = None if pm_T_months is None else pm_T_months.get(r.id, np.nan)
            if scheme == "PRW":
                Mi = np.interp(np.linspace(0, W, 200), self.t, self.M[i])
                fail = np.trapezoid(Mi, np.linspace(0, W, 200)) / W        # (1/W) int M
                cost += r.Cw * fail
            else:
                Th = None if (Tm is None or not np.isfinite(Tm)) else Tm * u
                fail = self._failures(i, W, Th)
                cost += r.Cw * fail
                if Th and g["pm_paid_by_maker"]:
                    pm_cost += r.Cpw_grp * math.floor(W / Th + 1e-12)
            claims += float(self.M_at(i, W)) if scheme == "PRW" else fail
        return dict(W=W_months, cost=cost + pm_cost, claims=claims)

    def warranty_breakdown(self, W_months):
        """Expected claims and cost per component at warranty length W (FRW, no PM)."""
        u, W = self.u, W_months * self.u
        rows = []
        for i, r in self.c.iterrows():
            f = float(self.M_at(i, W)) if r.warranted else 0.0
            rows.append(dict(id=r.id, component=r.component, beta=r.beta, claims=f, Cw=r.Cw, cost=f * r.Cw))
        return pd.DataFrame(rows)

    def premium(self, W):
        g = self.g
        return g["premium_cap"] * (1 - math.exp(-W / g["premium_tau"]))

    def warranty_optimise(self, pm_T_months=None, step=0.5):
        g = self.g
        grid = np.arange(g["w_min"], g["w_max"] + 1e-9, step)
        out = {}
        scen = {"FRW (no PM)": ("FRW", None), "PRW (no PM)": ("PRW", None)}
        if pm_T_months is not None:
            scen["FRW (with PM schedule)"] = ("FRW", pm_T_months)
        for name, (scheme, pmT) in scen.items():
            rows = []
            for W in grid:
                p = self.warranty_point(W, scheme, pmT)
                price = g["base_price"] + self.premium(W)
                profit = price - g["unit_cost"] - p["cost"]
                feas = (p["cost"] <= g["warranty_budget_frac"] * price) and (p["claims"] <= g["max_claims"])
                rows.append(dict(W=W, price=price, wcost=p["cost"], claims=p["claims"],
                                 profit=profit, feasible=feas))
            df = pd.DataFrame(rows)
            pool = df[df.feasible] if df.feasible.any() else df
            out[name] = dict(curve=df, best=pool.loc[pool.profit.idxmax()],
                             feasible_found=bool(df.feasible.any()))
        return out

    # ================================================== 4. validation (Monte-Carlo)
    def mc_validate(self, i, T_months, n=20000, seed=1):
        rng = np.random.default_rng(seed)
        r, u = self.c.loc[i], self.u
        T = T_months * u
        mu = r.mu
        K = int(6 * T / mu) + 25
        life = r.eta_h * rng.weibull(r.beta, size=(n, K))
        cnt = (np.cumsum(life, axis=1) <= T).sum(axis=1)
        M_mc, M_an = cnt.mean(), float(self.M_at(i, T))
        blk_mc = (r.Cp_ind + r.Cf * M_mc) / T * u
        blk_an = self.rate("block", i, T, r.Cp_ind, r.Cf) * u
        X = r.eta_h * rng.weibull(r.beta, size=n * 4)
        cyc = np.minimum(X, T)
        cost = np.where(X < T, r.Cf, r.Cp_ind)
        age_mc = cost.mean() / cyc.mean() * u
        age_an = self.rate("age", i, T, r.Cp_ind, r.Cf) * u
        return dict(M_analytic=M_an, M_mc=M_mc, M_err_pct=(M_mc / M_an - 1) * 100 if M_an else np.nan,
                    block_analytic=blk_an, block_mc=blk_mc,
                    age_analytic=age_an, age_mc=age_mc,
                    age_err_pct=(age_mc / age_an - 1) * 100)

    # ================================================== 5. explanation
    def explain_component(self, i, T_months=None):
        """Step-by-step, number-by-number derivation for one component."""
        r, g, u = self.c.loc[i], self.g, self.u
        ind = self.individual_policies().loc[i]
        T = (T_months if T_months else (ind.blk_T_mo)) * u
        L, D, Sf, hs = g["labour_rate"], g["downtime_cost"], g["setup_fixed"], g["setup_hours"]
        s = []
        s.append(("1. Reliability model",
                  f"Weibull(beta={r.beta}, eta={r.eta_h:,.0f} h).  MTTF = eta*Gamma(1+1/beta) = "
                  f"{r.eta_h:,.0f} x Gamma({1+1/r.beta:.3f}) = {r.mu:,.0f} h = {r.mu/u:.1f} months."))
        s.append(("2. Cost of a PLANNED replacement (stand-alone)",
                  f"Cp = part + (labour+downtime)x(t_p+setup_h) + setup_fixed = {r.part:,.0f} + ({L:,.0f}+{D:,.0f})x({r.t_p:g}+{hs:g}) + {Sf:,.0f} = Rs {r.Cp_ind:,.0f}"))
        s.append(("3. Cost of a FAILURE",
                  f"Cf = part + collateral + (labour+downtime)x(t_f+setup_h) + setup_fixed = {r.part:,.0f} + {r.collateral:,.0f} + ({L:,.0f}+{D:,.0f})x({r.t_f:g}+{hs:g}) + {Sf:,.0f} = Rs {r.Cf:,.0f}   (Cf/Cp = {r.Cf/r.Cp_ind:.2f})"))
        s.append(("4. Run-to-failure",
                  f"C_rtf = Cf/MTTF = {r.Cf:,.0f}/{r.mu:,.0f} = Rs {r.Cf/r.mu:.2f}/h = Rs {r.Cf/r.mu*u:,.0f}/month"))
        Mt, Ft, Rt, It = self.M_at(i, T), self.F_at(i, T), self.R_at(i, T), self.I_at(i, T)
        s.append((f"5. Evaluate both PM policies at T = {T/u:.1f} months ({T:,.0f} h)",
                  f"F(T) = 1-exp[-({T:,.0f}/{r.eta_h:,.0f})^{r.beta}] = {Ft:.4f};  R(T) = {Rt:.4f};  int_0^T R dx = {It:,.0f} h;  M(T) = {Mt:.4f} expected failures per interval (renewal eq.)"))
        age = (r.Cp_ind * Rt + r.Cf * Ft) / It
        blk = (r.Cp_ind + r.Cf * Mt) / T
        s.append(("6. Age-based PM  C(T) = [Cp R(T) + Cf F(T)] / int R",
                  f"= [{r.Cp_ind:,.0f}x{Rt:.4f} + {r.Cf:,.0f}x{Ft:.4f}] / {It:,.0f} = Rs {age:.2f}/h = Rs {age*u:,.0f}/month"))
        s.append(("7. Time-based (block) PM  C(T) = [Cp + Cf M(T)] / T",
                  f"= [{r.Cp_ind:,.0f} + {r.Cf:,.0f}x{Mt:.4f}] / {T:,.0f} = Rs {blk:.2f}/h = Rs {blk*u:,.0f}/month"))
        s.append(("8. Optimum (numerical search on the grid)",
                  f"Age-based T*={ind.age_T_mo:.1f} mo -> Rs {ind.age_rate_pm:,.0f}/mo;  Time-based T*={ind.blk_T_mo:.1f} mo -> Rs {ind.blk_rate_pm:,.0f}/mo;  RTF Rs {ind.rtf_rate_pm:,.0f}/mo"))
        s.append(("9. Decision & valid reason", f"{ind.policy}.  {ind.reason}"))
        s.append(("10. Warranty link",
                  f"Cw (claim cost to manufacturer) = Rs {r.Cw:,.0f}; expected claims in W=24 mo: M(W)={self.M_at(i,24*u):.3f} -> expected cost Rs {r.Cw*self.M_at(i,24*u):,.0f}"))
        return s


# ================================================================= pipeline
def run_pipeline(comps: pd.DataFrame, g: dict, schedule_basis: str = "recommended"):
    """schedule_basis: 'recommended' (practical calendar T0) or 'best' (global optimum T0)
    -- the PM schedule used for the 'FRW (with PM schedule)' warranty scenario."""
    if schedule_basis not in ("recommended", "best"):
        raise ValueError("schedule_basis must be 'recommended' or 'best'")
    eng = Engine(comps, g)
    ind = eng.individual_policies()
    grp = eng.group_summary(ind)
    tab = grp[schedule_basis]["table"]
    sched = dict(zip(tab.id, tab.T_mo))
    war = eng.warranty_optimise(pm_T_months=sched)
    return dict(engine=eng, individual=ind, group=grp, warranty=war, schedule=sched,
                schedule_basis=schedule_basis)


def headline(res):
    """Scalar KPIs used by the sensitivity analysis."""
    w = res["warranty"]
    k = "FRW (with PM schedule)"
    return dict(grouped_cost_pm=res["group"]["grouped_pm"],
                independent_cost_pm=res["group"]["independent_pm"],
                T0_opt=res["group"]["best"]["T0"],
                W_opt_noPM=w["FRW (no PM)"]["best"].W,
                profit_noPM=w["FRW (no PM)"]["best"].profit,
                W_opt_PM=w[k]["best"].W, profit_PM=w[k]["best"].profit)


def sensitivity(comps, g, delta=0.20):
    base = headline(run_pipeline(comps, g))
    specs = [
        ("Weibull scale eta (all)", lambda c, gg, f: c.assign(eta_h=c.eta_h * f)),
        ("Weibull shape beta (all)", lambda c, gg, f: c.assign(beta=c.beta * f)),
        ("Part prices", lambda c, gg, f: c.assign(part=c.part * f)),
        ("Secondary-damage cost", lambda c, gg, f: c.assign(collateral=c.collateral * f)),
        ("Corrective repair time t_f", lambda c, gg, f: c.assign(t_f=c.t_f * f)),
        ("Planned repair time t_p", lambda c, gg, f: c.assign(t_p=c.t_p * f)),
        ("Downtime cost Rs/h", lambda c, gg, f: (c, {**gg, "downtime_cost": gg["downtime_cost"] * f})),
        ("Visit setup cost (fixed)", lambda c, gg, f: (c, {**gg, "setup_fixed": gg["setup_fixed"] * f})),
        ("Visit setup hours", lambda c, gg, f: (c, {**gg, "setup_hours": gg["setup_hours"] * f})),
        ("Labour rate", lambda c, gg, f: (c, {**gg, "labour_rate": gg["labour_rate"] * f})),
        ("Warranty premium cap", lambda c, gg, f: (c, {**gg, "premium_cap": gg["premium_cap"] * f})),
    ]
    rows = []
    for name, fn in specs:
        out = {}
        for tag, f in (("lo", 1 - delta), ("hi", 1 + delta)):
            o = fn(comps, g, f)
            c2, g2 = (o if isinstance(o, tuple) else (o, g))
            out[tag] = headline(run_pipeline(c2, g2))
        rows.append(dict(parameter=name,
                         cost_lo=out["lo"]["grouped_cost_pm"], cost_hi=out["hi"]["grouped_cost_pm"],
                         T0_lo=out["lo"]["T0_opt"], T0_hi=out["hi"]["T0_opt"],
                         W_lo=out["lo"]["W_opt_PM"], W_hi=out["hi"]["W_opt_PM"],
                         profit_lo=out["lo"]["profit_PM"], profit_hi=out["hi"]["profit_PM"]))
    return base, pd.DataFrame(rows)


def constraint_sweeps(comps, g, r_list=(0.80, 0.90, 0.95, 0.98, 0.99), a_list=(0.980, 0.990, 0.993, 0.995, 0.996)):
    """Price of the reliability (R_min) and availability (A_min) constraints."""
    rows_r, rows_a = [], []
    for r in r_list:
        res = run_pipeline(comps, {**g, "r_min": r})
        b = res["group"]["best"]
        rows_r.append(dict(r_min=r, cost_pm=b["cost_pm"], T0=b["T0"], avail=b["avail"]))
    for a in a_list:
        res = run_pipeline(comps, {**g, "a_min": a})
        b = res["group"]["best"]
        rows_a.append(dict(a_min=a, cost_pm=b["cost_pm"], T0=b["T0"], avail=b["avail"], feasible=b["feasible"]))
    return pd.DataFrame(rows_r), pd.DataFrame(rows_a)


def group_plan(eng: "Engine", ind: pd.DataFrame, ev: dict):
    """Turn one eval_group() result into (a) per-component plan with a valid
    reason and (b) the list of maintenance groups (same interval).
    Planned repair hours of components sharing a shutdown are SUMMED (single crew,
    sequential work) -- a conservative duration assumption."""
    c, u, g = eng.c, eng.u, eng.g
    tab = ev["table"].copy()
    rows = []
    for i, r in tab.iterrows():
        cr = c.loc[i]
        rtf_pm = cr.Cf / cr.mu * u
        rate_pm = r.rate * u
        sav = (rtf_pm - rate_pm) / rtf_pm * 100
        if r.k == 0:
            base = ind.loc[i, "reason"]
            why = f"Run-to-failure. {base}" if "Run-to-failure" not in base else base
        else:
            T = r.T_mo
            Rt = float(eng.R_at(i, T * u))
            cp_drop = (1 - cr.Cp_grp / cr.Cp_ind) * 100
            if cr.critical and cr.beta > 1.05:
                why = (f"Safety-critical, beta={cr.beta:.1f}: replace every {T:.0f} mo so that R(T)={Rt:.3f} "
                       f"(>= {g['r_min']:.2f}); RTF not permitted. Shared visit cuts planned cost by {cp_drop:.0f}%.")
            else:
                why = (f"Wear-out beta={cr.beta:.1f}, Cf/Cp(shared)={cr.Cf/cr.Cp_grp:.1f}: replace every {T:.0f} mo "
                       f"(R(T)={Rt:.2f}); sharing the visit cuts planned cost {cp_drop:.0f}% and saves {sav:.1f}% vs RTF.")
        rows.append(dict(id=r.id, component=r.component, policy=("Run-to-failure" if r.k == 0 else "Group time-based PM"),
                         k=int(r.k), interval_mo=r.T_mo, cost_pm=rate_pm, rtf_pm=rtf_pm, saving_pct=sav,
                         downtime_h_per_1000h=r["dt"] * 1000, reason=why))
    plan = pd.DataFrame(rows)
    groups = []
    for k, grp in plan[plan.k > 0].groupby("k"):
        T = ev["T0"] * k
        ids = list(grp.id)
        cost_event = float(sum(c.loc[c.id == x, "Cp_grp"].iloc[0] for x in ids) + eng.S_visit)
        down_event = float(sum(c.loc[c.id == x, "t_p"].iloc[0] for x in ids) + eng.D_visit)
        groups.append(dict(group=f"G{len(groups)+1}", interval_months=T, events_per_year=12 / T,
                           components=", ".join(grp.component), ids=", ".join(ids),
                           cost_per_event=cost_event, downtime_h_per_event=down_event))
    groups = pd.DataFrame(groups).sort_values("interval_months").reset_index(drop=True)
    if len(groups):
        groups["group"] = [f"G{j+1}" for j in range(len(groups))]
    return plan, groups


def export_tables(res: dict, g: dict, extra: dict | None = None) -> dict:
    """All result tables as {sheet_name: DataFrame} (sheet names <= 31 chars) for
    Excel / CSV export. `extra` holds additional tables (e.g. failure modes)."""
    eng, grp = res["engine"], res["group"]
    drop_internal = lambda d: d.drop(columns=[c for c in d.columns if c.startswith("_")])
    plan_r, groups_r = group_plan(eng, res["individual"], grp["recommended"])
    plan_b, groups_b = group_plan(eng, res["individual"], grp["best"])
    out = {
        "Baseline parameters": pd.DataFrame({"parameter": list(g.keys()), "value": [str(v) for v in g.values()]}),
        "Components": eng.c.drop(columns=["Tmax_h"]),
        "Individual policies": drop_internal(res["individual"]),
        "Group plan (recommended)": plan_r,
        "Groups (recommended)": groups_r,
        "Group plan (optimum)": plan_b,
        "Groups (optimum)": groups_b,
        "T0 sweep": grp["sweep"],
        "Practical T0": grp["practical"],
    }
    for name, v in res["warranty"].items():
        out["Warranty " + name.replace("(", "").replace(")", "")[:21]] = v["curve"]
    if extra:
        out.update(extra)
    return {k[:31]: v for k, v in out.items()}
