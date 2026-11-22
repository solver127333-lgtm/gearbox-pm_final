"""
data.py -- Reference system definition and baseline input data
==============================================================
System : two-stage helical industrial gearbox, ~75 kW, 1500 rpm input,
         continuous conveyor duty (600 operating hours per month).

Units  : time = operating hours, money = INR (Rs), repair times = hours.

Parameter classes
-----------------
1. LITERATURE / OEM-DERIVED reliability parameters (Weibull shape beta and a
   reference life that is converted to the scale eta). Each row of
   DEFAULT_COMPONENTS carries a `life_basis` text and `source` keys that point
   into REFERENCES.
2. SITE / COMMERCIAL BASELINE parameters (prices, downtime cost, labour rate,
   repair durations, warranty economics). These are plant- and market-specific,
   so they are configurable planning baselines with documented typical ranges
   (BASELINE_PARAMETERS), not universal constants.

Life -> scale conversion (two-parameter Weibull)
------------------------------------------------
A reference life L_p at failure probability p (p = 0.10 for an L10 life) gives

        eta = L_p / [ -ln(1 - p) ] ** (1 / beta)

so L10 = eta * (-ln 0.90)^(1/beta) = 0.1054^(1/beta) * eta.   (Zaretsky 2016)
Using a two-parameter form (no failure-free interval) is conservative for gears
and shafts, for which a three-parameter form is often used (Beermann 2018).
"""
import math
import pandas as pd


def eta_from_life(life_h: float, p: float, beta: float) -> float:
    """Weibull scale (h) from a reference life `life_h` at failure probability `p`."""
    return life_h / (-math.log(1.0 - p)) ** (1.0 / beta)


# --------------------------------------------------------------------------
# 1. References (only sources actually used for a parameter or a table entry)
# --------------------------------------------------------------------------
REFERENCES = {
    "R1": dict(
        citation="Zaretsky, E. V. (2016). Rolling Bearing Life Prediction, Theory, and Application. "
                 "NASA/TP-2013-215305/REV1, NASA Glenn Research Center.",
        supports="Weibull slope of rolling bearings: 1.11 (point contact), 1.125 (line contact), "
                 "usual range 1-2; L10 = eta*(-ln 0.9)^(1/beta)."),
    "R2": dict(
        citation="Beermann, S. (2018). Reliability, Lifetime and Safety Factors. Gear Technology, "
                 "March/April 2018, pp. 58-65, Table 2 (after Bertsche).",
        supports="Weibull shape b: tooth flank (pitting) 1.1-1.5; tooth root (bending) 1.2-2.2; "
                 "shafts 1.1-1.9; ball bearing 1.1; roller bearing 1.35. ISO 281 a1 = 1.0 at 90 % "
                 "and 0.25 at 99 %; AGMA 2001 K_R = 0.85 (90 %), 1.00 (99 %)."),
    "R3": dict(
        citation="Gabelli, A., Doyer, A., Morales-Espejel, G. (2015). The Modified Life Rating of Rolling "
                 "Bearings: A Criterion for Gearbox Design and Reliability Optimization. Power "
                 "Transmission Engineering, March 2015, pp. 46-54.",
        supports="Worked 66 kW three-stage helical gear unit (1400 rpm): basic L10h 34,000 h (input taper "
                 "roller bearing), 24,000 h (intermediate shaft), 60,500 h (output shaft); modified life "
                 "8,500-29,000 h where lubrication/contamination limits apply."),
    "R4": dict(
        citation="Rinaldo, J. M. (2018). A Comparison of Current AGMA, ISO and API Gear Rating Methods. "
                 "Gear Technology, July 2018, pp. 46-52.",
        supports="Gear ratings assume 20 years of continuous operation (175,200 h); ANSI/AGMA 6011 "
                 "ratings are for a minimum of 40,000 h; ANSI/AGMA 6013 rating is for 10,000 operating hours."),
    "R5": dict(
        citation="Flender GmbH. Assembly and operating instructions BA 7300, Gear unit lubrication and "
                 "gear unit preservation, version 07/2026.",
        supports="Oil service life at 80 C sump: mineral oil/synthetic ester 10,000 h (max 2 years); "
                 "PAO/polyglycol 20,000 h (max 4 years); +10 K halves life (Arrhenius rule). Rotary shaft "
                 "seals expected life: >=5 years (NBR, mineral oil), >=2 years (NBR, synthetic), up to 10 "
                 "years (FKM). Wet-air (desiccant) filter replacement criteria."),
    "R6": dict(
        citation="SEW-EURODRIVE GmbH & Co KG. Operating instructions, X..e series helical and bevel-helical "
                 "gear units, sections 'Inspection and maintenance intervals' and 'Lubricant change "
                 "intervals' (online edition 04/2026).",
        supports="Breather replacement recommended once a year; monthly gear-unit leakage check; "
                 "seals replaced when damaged; housing temperature limits 90 C (mineral) / 100 C "
                 "(synthetic); oil analysis recommended to optimise change intervals."),
    "R7": dict(
        citation="ISO 15243:2017. Rolling bearings - Damage and failures - Terms, characteristics and causes.",
        supports="Failure-mode classes: rolling contact fatigue (sub-/surface-initiated), wear (abrasive/"
                 "adhesive), corrosion (moisture, fretting, false brinelling), electrical erosion, plastic "
                 "deformation, cracking and fracture."),
    "R8": dict(
        citation="ISO 10825:1995 (revised by ISO 10825-1:2022). Gears - Wear and damage to gear teeth - "
                 "Terminology.",
        supports="Gear damage classes: surface disturbances (wear, corrosion, overheating), scuffing, permanent "
                 "deformation, surface fatigue (pitting, micropitting, spalling), cracks, tooth breakage."),
    "R9": dict(
        citation="NREL Gearbox Reliability Database - 2014 Wind Turbine Gearbox Damage Distribution "
                 "(U.S. DOE National Renewable Energy Laboratory, published Feb 2015; ~320 damage records).",
        supports="Share of gearbox damage records: bearings 64 %, gears 25 %, other 11 % (wind-turbine "
                 "gearboxes; context for failure-mode emphasis, not a model input)."),
    "R10": dict(
        citation="NSK Ltd. Industrial Gearboxes (brochure ASB/25, 2025).",
        supports="About 40 % of premature rolling-bearing damage attributed to improper lubrication; "
                 "contamination and poor handling during assembly or maintenance as common causes."),
    "R11": dict(
        citation="Sumitomo Drive Technologies. 3 most common gearbox failures (web article).",
        supports="Premature bearing failure stated to cause more than 50 % of all gearbox failures; gear "
                 "failures linked to contaminated oil, misalignment, heat; seal failures linked to contamination."),
    "R12": dict(
        citation="Freudenberg Sealing Technologies (FST). Radial shaft seals (product information).",
        supports="Service life up to 20,000 h quoted for the MSS 1 radial shaft seal design; lip/hollow wear "
                 "as wear mechanisms."),
    "R13": dict(
        citation="Wright, D. (2005). Spur gears: failure and reliability. University of Western Australia, "
                 "DANotes (simplified AGMA 2001).",
        supports="Gear tooth failure modes (bending fatigue, flank pitting, scuffing); AGMA 2001 reliability "
                 "factor K_R 0.85 at 90 %, 1.00 at 99 %."),
    "R14": dict(
        citation="ISO 281:2007. Rolling bearings - Dynamic load ratings and rating life.",
        supports="Definition of the L10 rating life (90 % reliability) used to convert bearing lives to eta."),
    "R15": dict(
        citation="Barlow, R. E., Proschan, F. (1965). Mathematical Theory of Reliability. Wiley.",
        supports="Age-replacement cost-rate formula; renewal theory."),
    "R16": dict(
        citation="Blischke, W. R., Murthy, D. N. P. (1994). Warranty Cost Analysis. Marcel Dekker.",
        supports="Free-replacement (FRW) and pro-rata (PRW) warranty cost models."),
    "R17": dict(
        citation="Dekker, R., Wildeman, R. E., van der Duyn Schouten, F. A. (1997). A review of "
                 "multi-component maintenance models with economic dependence. Mathematical Methods of "
                 "Operations Research 45, 411-435.",
        supports="Group maintenance with shared set-up cost."),
    "R18": dict(
        citation="Xie, M. (1989). On the solution of renewal-type integral equations. Communications in "
                 "Statistics - Simulation and Computation 18(1), 281-293.",
        supports="Numerical solution of the renewal equation."),
}

# --------------------------------------------------------------------------
# 2. Site / commercial baseline parameters (configurable planning values)
# --------------------------------------------------------------------------
DEFAULT_GLOBALS = dict(
    # operating profile
    hours_per_month=600.0,
    # cost drivers
    labour_rate=2000.0,
    downtime_cost=12000.0,
    setup_fixed=15000.0,
    setup_hours=3.0,
    # constraints
    r_min=0.90,
    a_min=0.990,
    budget_per_year=750_000.0,
    pm_saving_threshold=0.02,
    t_min_months=1.0,
    k_max=24,
    # warranty / commercial
    base_price=1_500_000.0,
    unit_cost=1_050_000.0,
    premium_cap=150_000.0,
    premium_tau=18.0,
    w_min=6.0, w_max=60.0,
    warranty_budget_frac=0.08,
    max_claims=0.60,
    warranty_covers_downtime=False,
    pm_paid_by_maker=False,
)

# Documentation of every baseline parameter (shown in the app)
BASELINE_PARAMETERS = [
    ("hours_per_month", "Operating hours per month", "h", "300-720",
     "20 h/day x 30 days continuous conveyor duty; 720 = 24/7."),
    ("labour_rate", "Maintenance labour rate (loaded crew)", "Rs/h", "800-3,000",
     "Two-technician crew incl. overhead; plant-specific."),
    ("downtime_cost", "Production loss while the gearbox is down", "Rs/h", "5,000-50,000",
     "Conveyor-fed process; set from the plant's contribution margin per hour."),
    ("setup_fixed", "Fixed cost per maintenance visit", "Rs", "5,000-50,000",
     "Crane/hoist, scaffolding, lock-out/tag-out, oil handling, travel; paid once per shutdown."),
    ("setup_hours", "Shutdown hours per visit (isolate, drain, open, refill, run-up)", "h", "1-8",
     "Common to all tasks in a visit; component repair times are additional."),
    ("r_min", "Minimum reliability at replacement, safety-critical parts", "-", "0.80-0.99",
     "0.90 aligns with the ISO 281 / L10 convention (90 % reliability)."),
    ("a_min", "Minimum system availability", "-", "0.95-0.999", "Maintainability requirement of the process."),
    ("budget_per_year", "Maximum annual maintenance budget for this unit", "Rs/yr", "site-specific",
     "Cap on expected planned + corrective cost incl. downtime."),
    ("pm_saving_threshold", "Minimum economic saving for a PM requirement", "-", "0-0.10",
     "A planned replacement must beat run-to-failure by at least this fraction."),
    ("t_min_months", "Shortest practical PM interval", "months", "0.5-6", "Planning/shutdown practicality."),
    ("k_max", "Largest multiple of the base interval used in group PM", "-", "4-24",
     "Allowed multiples are those of (1,2,3,4,6,8,12,16,24) not exceeding k_max."),
    ("base_price", "List price at zero warranty", "Rs", "market-specific", "Commercial baseline."),
    ("unit_cost", "Production + distribution cost per unit", "Rs", "market-specific", "Commercial baseline."),
    ("premium_cap", "Maximum price premium customers accept for a long warranty", "Rs", "5-15 % of price",
     "Saturating willingness-to-pay curve: cap*(1-exp(-W/tau))."),
    ("premium_tau", "Premium saturation time", "months", "12-36", "Speed at which extra warranty stops adding value."),
    ("w_min / w_max", "Allowed warranty length", "months", "6-120", "Market / policy limits."),
    ("warranty_budget_frac", "Maximum expected warranty cost as share of price", "-", "0.02-0.10",
     "Warranty reserve limit."),
    ("max_claims", "Maximum expected warranty claims per unit", "-", "0.2-1.0", "Service-capacity limit."),
]

# --------------------------------------------------------------------------
# 3. Component table
#    beta   : Weibull shape (see `source`)
#    life_ref_h, life_p : reference life [h] at failure probability p
#    eta_h  : Weibull scale derived from the reference life (formula above)
#    part, collateral : part price and secondary-damage cost on in-service failure (Rs)
#    t_p, t_f : planned replacement / corrective repair hours, in ADDITION to the
#               shared shutdown hours of a visit (diagnosis, spares wait, repair)
#    critical : consequence-critical wear-out part (run-to-failure not allowed)
#    warranted: failure claimable under warranty (False for lubricant: routine duty)
# --------------------------------------------------------------------------
_COLS = ["id", "assembly", "sub_assembly", "component", "beta", "life_ref_h", "life_p",
         "part", "collateral", "t_p", "t_f", "critical", "warranted", "life_basis", "source"]
_ROWS = [
 ("C1", "Input (HS) shaft assy", "Pinion", "HS pinion shaft (gear cut integral)",
  1.3, 40000, 0.10, 120000, 200000, 20, 48, True, True,
  "Flank pitting b=1.3 (mid of 1.1-1.5); rated life 40,000 h taken as L10", "R2, R4, R13"),
 ("C2", "Input (HS) shaft assy", "HS support", "HS bearing set (taper roller)",
  1.35, 34000, 0.10, 35000, 60000, 6, 14, False, True,
  "Roller bearing b=1.35; basic L10h 34,000 h (66 kW helical unit, input taper roller)", "R2, R3, R14"),
 ("C3", "Input (HS) shaft assy", "HS sealing", "HS oil seal (NBR)",
  2.0, 36000, 0.10, 3000, 10000, 2, 4, False, True,
  ">=5 years NBR/mineral = 36,000 h at 600 h/month taken as L10; b=2.0 baseline (ageing wear-out)", "R5, R12"),
 ("C4", "Intermediate shaft assy", "Gear shaft", "Intermediate gear + shaft",
  1.3, 40000, 0.10, 180000, 250000, 24, 56, True, True,
  "Flank pitting b=1.3; rated life 40,000 h taken as L10", "R2, R4, R13"),
 ("C5", "Intermediate shaft assy", "Support", "Intermediate bearing set (taper roller)",
  1.35, 24000, 0.10, 40000, 70000, 7, 16, False, True,
  "Roller bearing b=1.35; basic L10h 24,000 h (66 kW helical unit, intermediate shaft)", "R2, R3, R14"),
 ("C6", "Output (LS) shaft assy", "Gear", "LS gear wheel",
  1.3, 40000, 0.10, 250000, 300000, 28, 64, True, True,
  "Flank pitting b=1.3; rated life 40,000 h taken as L10", "R2, R4, R13"),
 ("C7", "Output (LS) shaft assy", "Shaft", "LS output shaft + key",
  1.5, 175200, 0.10, 140000, 150000, 22, 50, False, True,
  "Shaft b=1.5 (mid of 1.1-1.9); 20-year continuous rating 175,200 h taken as L10", "R2, R4"),
 ("C8", "Output (LS) shaft assy", "Support", "LS bearing set (spherical roller)",
  1.35, 60500, 0.10, 70000, 100000, 8, 18, False, True,
  "Roller bearing b=1.35; basic L10h 60,500 h (66 kW helical unit, output shaft)", "R2, R3, R14"),
 ("C9", "Output (LS) shaft assy", "Sealing", "LS oil seal (NBR)",
  2.0, 36000, 0.10, 4000, 10000, 2, 4, False, True,
  ">=5 years NBR/mineral = 36,000 h at 600 h/month taken as L10; b=2.0 baseline (ageing wear-out)", "R5, R12"),
 ("C10", "Lubrication & cooling", "Lubricant", "Gear oil (mineral CLP) + filter",
  3.5, 10000, 0.10, 65000, 250000, 2, 6, False, False,
  "OEM service life 10,000 h at 80 C taken as L10; b=3.5 baseline (sharp ageing onset)", "R5, R6"),
 ("C11", "Lubrication & cooling", "Breathing", "Breather / air filter",
  2.5, 7200, 0.10, 4000, 20000, 0.5, 2, False, True,
  "OEM: replace once a year = 7,200 h at 600 h/month taken as L10; b=2.5 baseline (desiccant saturation)", "R5, R6"),
 ("C12", "Casing", "Housing", "Housing + fasteners",
  1.0, 600000, None, 350000, 100000, 40, 80, False, True,
  "Random overload events: b=1 (constant hazard), eta=600,000 h baseline (rare)", "Baseline"),
 ("C13", "Installation & assembly", "Build quality", "Assembly/installation defects (alignment, torque, fits)",
  0.7, 400000, None, 15000, 50000, 4, 16, False, True,
  "Early-life (infant-mortality) model b=0.7, eta=400,000 h baseline; calibrate to commissioning records",
  "R10, Baseline"),
]
DEFAULT_COMPONENTS = pd.DataFrame(_ROWS, columns=_COLS)
DEFAULT_COMPONENTS["eta_h"] = [
    eta_from_life(L, p, b) if pd.notna(p) else float(L)
    for L, p, b in zip(DEFAULT_COMPONENTS.life_ref_h, DEFAULT_COMPONENTS.life_p, DEFAULT_COMPONENTS.beta)]
DEFAULT_COMPONENTS["eta_h"] = DEFAULT_COMPONENTS["eta_h"].round(0)
DEFAULT_COMPONENTS = DEFAULT_COMPONENTS[
    ["id", "assembly", "sub_assembly", "component", "beta", "eta_h", "part", "collateral",
     "t_p", "t_f", "critical", "warranted", "life_ref_h", "life_p", "life_basis", "source"]]

# --------------------------------------------------------------------------
# 4. Failure modes, causes and mechanisms (classes from ISO 15243 / ISO 10825)
# --------------------------------------------------------------------------
_FM_COLS = ["id", "component", "failure_mode", "failure_cause", "failure_mechanism", "source"]
_GEAR_FM = [
    ("Pitting / micropitting (surface fatigue)", "Overload, misalignment, inadequate or contaminated lubricant film",
     "Rolling-contact (Hertzian) surface fatigue", "R8, R13"),
    ("Tooth breakage (fatigue breakage)", "Overload, shock loads, misalignment, root stress concentration",
     "Bending fatigue at the tooth root", "R8, R13"),
    ("Scuffing", "Thin lubricant film at high speed, over-temperature",
     "Local surface welding and material transfer (adhesive)", "R8, R13"),
]
_BRG_FM = [
    ("Subsurface-initiated fatigue (spalling)", "High load, misalignment, contamination",
     "Rolling-contact fatigue (L10 rating-life basis)", "R7, R14"),
    ("Wear / surface-initiated fatigue", "Improper lubrication (about 40 % of premature damage), contamination",
     "Abrasive and adhesive wear, surface-initiated fatigue", "R7, R10"),
]
_FM_ROWS = []
for cid, nm in (("C1", "HS pinion shaft"), ("C4", "Intermediate gear + shaft"), ("C6", "LS gear wheel")):
    _FM_ROWS += [(cid, nm, *r) for r in _GEAR_FM]
for cid, nm in (("C2", "HS bearing set"), ("C5", "Intermediate bearing set"), ("C8", "LS bearing set")):
    _FM_ROWS += [(cid, nm, *r) for r in _BRG_FM]
_FM_ROWS += [
    ("C3", "HS oil seal", "Lip wear and oil leakage", "Contamination, shaft surface damage, heat",
     "Abrasive lip wear, elastomer hardening/ageing", "R5, R11, R12"),
    ("C9", "LS oil seal", "Lip wear and oil leakage", "Shaft run-out, contamination, heat",
     "Abrasive lip wear, elastomer hardening/ageing", "R5, R11, R12"),
    ("C7", "LS output shaft + key", "Fatigue fracture at keyway/shoulder", "Torque reversals, stress concentration",
     "High-cycle fatigue crack growth (shaft b=1.1-1.9)", "R2"),
    ("C10", "Gear oil + filter", "Oxidation / viscosity loss / contamination", "Sump temperature, water ingress, wear debris",
     "Thermal-oxidative ageing (life halves per +10 K); depleted oil accelerates all wear", "R5, R6"),
    ("C11", "Breather / air filter", "Desiccant saturation / clogging", "Humid or dusty environment",
     "Moisture and dust ingress into the sump", "R5, R6"),
    ("C12", "Housing + fasteners", "Casing crack / loosened fasteners", "Random overload, corrosion, vibration",
     "Random (memoryless) overload event", "Baseline"),
    ("C13", "Assembly/installation defects", "Early-life misalignment / wrong preload / contamination",
     "Poor handling and fits during assembly or maintenance", "Latent defect revealed early (decreasing hazard)", "R10"),
]
FAILURE_MODES = pd.DataFrame(_FM_ROWS, columns=_FM_COLS)

PLANNING_NOTE = (
    "Reliability parameters (beta, eta) are derived from the published shapes and reference lives in REFERENCES. "
    "Prices, repair durations, downtime cost and warranty economics are configurable site/commercial baselines; "
    "replace them with plant records and supplier quotations before using results for a decision. "
    "Gear/shaft lives use a two-parameter Weibull form, which is conservative relative to the three-parameter "
    "form (failure-free interval) used in the literature."
)
