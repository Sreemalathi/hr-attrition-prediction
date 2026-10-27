#!/usr/bin/env python3
"""
Synthetic HR dataset generator  (HR Dataset v14 style, built for time-based analysis)
====================================================================================
Simulated synthetic data. Nothing here describes real people.

Coverage
    Employees active at any time between 2021-01-01 and 2026-09-30 (the data cut-off).
    450 "legacy" employees were hired before 2021 and still employed on 2021-01-01.
    1,150 further employees were hired between Jan 2021 and Sep 2026.
    Total = 1,600 rows, one row per person.

Patterns deliberately built in (so you can find them, and so you know they are planted)
    - Early exits       : exit risk is highest in the first year of employment
    - Seasonality       : exits peak Feb-Mar (after bonus season), dip in December
    - Hiring freeze     : hiring slows sharply Jul-Dec 2023
    - Sales reorg       : Sales exit risk is 3x normal Jan-Jun 2024
    - IT/IS poaching    : IT/IS exit risk is 3.5x normal Oct-Dec 2025
    - Leaky department  : Sales has the highest baseline risk (plus heavy overtime)
    - Three high-attrition managers (see BAD_MANAGERS)
    - Individual drivers: overtime, low engagement, below-band pay, long time without
                          promotion, recruitment source; performance drives involuntary exits
    - Gender and DateofBirth (age) have NO effect on exits. They are included for diversity reporting.

Reproducible: same SEED -> same file.   Run:  python generate_hr_dataset.py
Edit the CONFIG block to change sizes, departments, events or the target attrition share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================ CONFIG
SEED = 42
N_TOTAL = 1600
N_LEGACY = 450
DATA_START = pd.Timestamp("2021-01-01")
CUTOFF = pd.Timestamp("2026-09-30")
TARGET_TERMINATED_SHARE = 0.38   # share of the 1,600 who have left by the cut-off
R_INVOL = 0.26                   # involuntary hazard relative to voluntary (tunes the cause/voluntary mix)

MONTHS = pd.period_range("2021-01", "2026-09", freq="M")
MONTH_STARTS = [p.start_time for p in MONTHS]

DEPTS = ["Production", "Sales", "IT/IS", "Software Engineering", "Admin Offices"]
DEPT_SHARE = [0.35, 0.20, 0.15, 0.18, 0.12]
DEPT_HAZARD = {"Production": 1.00, "Sales": 1.35, "IT/IS": 1.10,
               "Software Engineering": 1.05, "Admin Offices": 0.80}
FEMALE_SHARE = {"Production": 0.40, "Sales": 0.50, "IT/IS": 0.35,
                "Software Engineering": 0.30, "Admin Offices": 0.70}
OT_BASE = {"Production": 12, "Sales": 14, "IT/IS": 9, "Software Engineering": 10, "Admin Offices": 3}

POSITIONS = {
    "Production": ["Production Technician I", "Production Technician II",
                   "Senior Production Technician", "Production Lead"],
    "Sales": ["Sales Associate", "Account Executive", "Senior Account Executive", "Sales Team Lead"],
    "IT/IS": ["IT Support Analyst", "Systems Administrator", "Senior Systems Engineer", "IT Team Lead"],
    "Software Engineering": ["Junior Software Engineer", "Software Engineer",
                             "Senior Software Engineer", "Engineering Lead"],
    "Admin Offices": ["Administrative Assistant", "Office Coordinator",
                      "Operations Specialist", "Office Manager"],
}
BAND_MID = {  # annual salary mid-point in EUR by job level 1..4
    "Production": [36000, 42000, 50000, 62000],
    "Sales": [40000, 50000, 62000, 80000],
    "IT/IS": [48000, 58000, 70000, 88000],
    "Software Engineering": [55000, 68000, 82000, 100000],
    "Admin Offices": [36000, 42000, 50000, 64000],
}
MANAGERS = {
    "Production": [f"PRD-M{i}" for i in range(1, 6)],
    "Sales": [f"SLS-M{i}" for i in range(1, 5)],
    "IT/IS": [f"ITS-M{i}" for i in range(1, 4)],
    "Software Engineering": [f"SWE-M{i}" for i in range(1, 5)],
    "Admin Offices": [f"ADM-M{i}" for i in range(1, 4)],
}
BAD_MANAGERS = {"PRD-M4", "SLS-M2", "SWE-M3"}

SOURCES = ["LinkedIn", "Indeed", "Employee Referral", "Company Website",
           "Google Search", "Campus Recruiting", "Diversity Job Fair"]
SOURCE_P = [0.26, 0.22, 0.14, 0.12, 0.12, 0.08, 0.06]
SOURCE_MULT = {"LinkedIn": 1.00, "Indeed": 1.20, "Employee Referral": 0.60, "Company Website": 0.90,
               "Google Search": 1.25, "Campus Recruiting": 0.85, "Diversity Job Fair": 1.00}

HIRE_SEASON = {1: 1.2, 2: 1.2, 3: 1.2, 4: 1.0, 5: 1.0, 6: 0.9,
               7: 0.8, 8: 0.8, 9: 1.2, 10: 1.1, 11: 0.8, 12: 0.5}
EXIT_SEASON = {1: 1.15, 2: 1.45, 3: 1.35, 4: 1.0, 5: 0.95, 6: 0.9,
               7: 0.95, 8: 1.0, 9: 1.1, 10: 1.05, 11: 0.85, 12: 0.7}

VOL_REASONS = ["Better opportunity", "Better salary", "Dissatisfaction with management",
               "Work hours / overtime", "Career change", "Relocation", "Retirement"]
INV_REASONS = ["Performance", "Attendance", "Misconduct"]


# ============================================================================ HELPERS
def tenure_mult(months):
    """Exit risk by months of tenure: highest in year one, falling afterwards."""
    return np.select([months < 6, months < 12, months < 24, months < 48, months < 96],
                     [1.5, 2.0, 1.3, 1.0, 0.8], default=0.65)


def sample_weighted(weights, rng):
    """Pick one column per row from a (rows x categories) matrix of non-negative weights."""
    c = weights.cumsum(axis=1)
    u = rng.random(len(weights)) * c[:, -1]
    return (u[:, None] > c).sum(axis=1)


# ============================================================================ BUILD
def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = len(MONTHS)
    n = N_TOTAL
    n_new = N_TOTAL - N_LEGACY

    # ---- 1. hire dates ------------------------------------------------------------
    w = (1 + 0.004 * np.arange(T)) * np.array([HIRE_SEASON[p.month] for p in MONTHS])
    for i, p in enumerate(MONTHS):
        if pd.Period("2023-07", "M") <= p <= pd.Period("2023-12", "M"):
            w[i] *= 0.25                                   # hiring freeze
    counts = rng.multinomial(n_new, w / w.sum())
    new_dates = []
    for p, c in zip(MONTHS, counts):
        days = rng.integers(0, p.days_in_month, size=c)
        new_dates.extend(list(p.start_time + pd.to_timedelta(days, unit="D")))
    draws = rng.exponential(1100, N_LEGACY * 3)
    days_before = (draws[draws <= 3470][:N_LEGACY]).astype(int) + 30     # truncate by re-drawing, not clipping
    legacy_dates = DATA_START - pd.to_timedelta(days_before, unit="D")
    hire = legacy_dates.append(pd.DatetimeIndex(new_dates)).sort_values()
    hire64 = hire.values
    hire_idx = np.where(hire < DATA_START, -1, (hire.year - 2021) * 12 + hire.month - 1)
    legacy = np.asarray(hire < DATA_START)

    # ---- 2. static attributes -----------------------------------------------------
    dept = rng.choice(DEPTS, p=DEPT_SHARE, size=n)
    level = np.where(legacy,
                     rng.choice([1, 2, 3, 4], p=[0.20, 0.35, 0.30, 0.15], size=n),
                     rng.choice([1, 2, 3, 4], p=[0.38, 0.34, 0.20, 0.08], size=n))
    manager = np.empty(n, dtype=object)
    gender = np.empty(n, dtype=object)
    for d in DEPTS:
        m = dept == d
        manager[m] = rng.choice(MANAGERS[d], size=m.sum())
        f = FEMALE_SHARE[d]
        gender[m] = rng.choice(["Female", "Male", "Diverse"], p=[f, 1 - f - 0.01, 0.01], size=m.sum())
    bad_mgr = np.array([x in BAD_MANAGERS for x in manager])
    source = rng.choice(SOURCES, p=SOURCE_P, size=n)
    src_mult = np.array([SOURCE_MULT[s] for s in source])
    position = np.array([POSITIONS[d][l - 1] for d, l in zip(dept, level)])
    age_at_hire = np.clip(np.where(legacy, rng.normal(31, 7, n), rng.normal(33, 8, n))
                          + 2 * (level - 1), 20, 60).round().astype(int)

    # ---- 3. work-life measures ----------------------------------------------------
    ot_base = np.array([OT_BASE[d] for d in dept], dtype=float)
    overtime = np.clip(rng.gamma(2.5, ot_base / 2.5) + 5 * bad_mgr, 0, 45).round().astype(int)
    engagement = np.clip(3.9 - 0.45 * bad_mgr - 0.018 * (overtime - 8) + rng.normal(0, 0.5, n),
                         1.5, 5.0).round(2)
    z = 0.3 * (engagement - 3.8) + rng.normal(0, 1, n)
    q05, q15, q83 = np.quantile(z, [0.05, 0.15, 0.83])
    perf = np.select([z < q05, z < q15, z < q83],
                     ["PIP", "Needs Improvement", "Fully Meets"], default="Exceeds")
    satisfaction = np.clip(np.round(engagement + rng.normal(0, 0.7, n)), 1, 5).astype(int)
    projects = np.clip(rng.poisson(0.8 + 0.5 * (level - 1) + 1.0 * (perf == "Exceeds")), 0, 8)
    lam_late = 0.25 + 0.5 * np.isin(perf, ["PIP", "Needs Improvement"]) + 0.1 * (overtime > 25)
    days_late = np.clip(rng.poisson(lam_late), 0, 8)
    absences = np.clip(rng.poisson(6 + 0.05 * overtime), 0, 20)
    stagnant = rng.random(n) < 0.22                       # latent "no promotion for years"
    compa = np.clip(rng.normal(1.0, 0.08, n) + 0.03 * (perf == "Exceeds") - 0.02 * (perf == "Needs Improvement"),
                    0.75, 1.30)
    salary = (np.array([BAND_MID[d][l - 1] for d, l in zip(dept, level)]) * compa / 100).round() * 100

    # ---- 4. exit simulation (monthly hazards) -------------------------------------
    ln_v = (np.log(np.array([DEPT_HAZARD[d] for d in dept]))
            + 0.022 * overtime
            - 0.60 * (engagement - 3.6)
            + 0.60 * bad_mgr
            + np.log(src_mult)
            + 0.80 * (compa < 0.93) - 0.25 * (compa > 1.08))
    ln_i = (1.2 * (perf == "Needs Improvement") + 1.9 * (perf == "PIP") + 0.35 * (days_late >= 3))
    U1 = rng.random((n, T))
    U2 = rng.random((n, T))
    is_sales, is_its = dept == "Sales", dept == "IT/IS"

    def simulate(scale):
        alive = np.ones(n, bool)
        exit_idx = np.full(n, -1)
        invol = np.zeros(n, bool)
        for t in range(T):
            start = MONTH_STARTS[t]
            active = alive & (hire_idx < t)
            if not active.any():
                continue
            ten_m = (np.datetime64(start) - hire64) / np.timedelta64(1, "D") / 30.44
            event = np.ones(n)
            if pd.Timestamp("2024-01-01") <= start <= pd.Timestamp("2024-06-30"):
                event[is_sales] = 3.0
            if pd.Timestamp("2025-10-01") <= start <= pd.Timestamp("2025-12-31"):
                event[is_its] = 3.5
            stag = np.where((ten_m >= 36) & stagnant, np.exp(0.25), 1.0)
            hv = scale * np.exp(ln_v) * stag * tenure_mult(ten_m) * EXIT_SEASON[start.month] * event
            hi = scale * R_INVOL * np.exp(ln_i) * np.where(ten_m < 6, 1.5, 1.0)
            p = 1 - np.exp(-(hv + hi))
            ex = active & (U1[:, t] < p)
            exit_idx[ex] = t
            invol[ex] = (U2[:, t] < hi / (hv + hi))[ex]
            alive &= ~ex
        return exit_idx, invol

    lo, hi_ = 1e-4, 0.05
    for _ in range(40):                                   # calibrate to the target share of leavers
        mid = (lo + hi_) / 2
        ex_idx, _ = simulate(mid)
        if (ex_idx >= 0).mean() < TARGET_TERMINATED_SHARE:
            lo = mid
        else:
            hi_ = mid
    exit_idx, invol = simulate((lo + hi_) / 2)
    left = exit_idx >= 0

    # ---- 5. termination dates & reasons -------------------------------------------
    term = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    for i in np.where(left)[0]:
        p = MONTHS[exit_idx[i]]
        d = p.start_time + pd.Timedelta(days=int(rng.integers(0, p.days_in_month)))
        d = max(d, hire[i] + pd.Timedelta(days=14))
        term[i] = min(d, CUTOFF)
    end = term.fillna(CUTOFF)
    ten_years_end = ((end - pd.Series(hire)).dt.days / 365.25).to_numpy()

    reason = np.full(n, "", dtype=object)
    ev = np.where(left & ~invol)[0]
    if len(ev):
        age_exit = age_at_hire[ev] + ten_years_end[ev]
        wv = np.column_stack([
            1.2 + 1.5 * (stagnant[ev] & (ten_years_end[ev] >= 3)),
            0.8 + 3.0 * (compa[ev] < 0.93),
            0.4 + 2.5 * bad_mgr[ev],
            0.5 + 0.08 * np.maximum(overtime[ev] - 15, 0),
            np.full(len(ev), 0.5),
            np.full(len(ev), 0.4),
            5.0 * (age_exit >= 61),
        ])
        reason[ev] = np.array(VOL_REASONS)[sample_weighted(wv, rng)]
    ei = np.where(left & invol)[0]
    if len(ei):
        wi = np.column_stack([
            0.5 + 3.0 * np.isin(perf[ei], ["PIP", "Needs Improvement"]),
            0.5 + 2.0 * ((days_late[ei] >= 3) | (absences[ei] >= 12)),
            np.full(len(ei), 0.4),
        ])
        reason[ei] = np.array(INV_REASONS)[sample_weighted(wi, rng)]
    status = np.where(~left, "Active", np.where(invol, "Terminated for Cause", "Voluntary Resignation"))

    # ---- 6. years since last promotion --------------------------------------------
    yslp = np.where(stagnant & (ten_years_end >= 3), rng.uniform(3, 8, n), rng.uniform(0, 2.5, n))
    yslp = np.minimum(yslp, ten_years_end).round(1)

    # ---- 7. date of birth ---------------------------------------------------------
    # Consistent with the age on the hire date (completed years). A separate random
    # generator is used so that no other column changes when this one is added.
    dob_rng = np.random.default_rng(seed + 1)
    dob = pd.DatetimeIndex([h - pd.DateOffset(years=int(a)) - pd.Timedelta(days=int(d))
                            for h, a, d in zip(hire, age_at_hire, dob_rng.integers(0, 365, n))])

    df = pd.DataFrame({
        "EmpID": 10001 + np.arange(n),
        "Department": dept,
        "Position": position,
        "JobLevel": level,
        "ManagerID": manager,
        "RecruitmentSource": source,
        "Gender": gender,
        "DateofBirth": dob.strftime("%Y-%m-%d"),
        "Salary": salary.astype(int),
        "DateofHire": hire.strftime("%Y-%m-%d"),
        "DateofTermination": term.dt.strftime("%Y-%m-%d").fillna(""),
        "EmploymentStatus": status,
        "TermReason": reason,
        "PerformanceScore": perf,
        "EngagementSurvey": engagement,
        "EmpSatisfaction": satisfaction,
        "SpecialProjectsCount": projects,
        "DaysLateLast30": days_late,
        "Absences": absences,
        "OvertimeHoursPerMonth": overtime,
        "YearsSinceLastPromotion": yslp,
    })
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="hr_synthetic_dataset.csv")
    a = ap.parse_args()
    out = build(a.seed)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}")
    print(out["EmploymentStatus"].value_counts().to_string())
