# HR_dataset: Data Dictionary

**Simulated synthetic data.**

- **Rows:** 1,600, one per person (EmpID is unique)
- **Coverage:** everyone employed at any point from 2021-01-01 to the cut-off date **2026-09-30**
  - 450 people were hired before 2021 and still employed on 2021-01-01
  - 1,150 were hired between Jan 2021 and Sep 2026
- **Outcome mix:** 992 Active, 504 Voluntary Resignation, 104 Terminated for Cause (38% have left)
- **Reproducible:** `python generate_hr_dataset.py` recreates the same file (seed 42)

## Columns

| Column | Meaning | Use as a model feature? |
|---|---|---|
| EmpID | Unique employee ID | No (identifier) |
| Department | Production, Sales, IT/IS, Software Engineering, Admin Offices | Yes |
| Position | Job title (set by department and level) | Careful: overlaps with JobLevel |
| JobLevel | 1 (entry) to 4 (lead) | Yes |
| ManagerID | Team lead ID (managers are not rows in this table) | Yes |
| RecruitmentSource | Channel the person was hired through | Yes |
| Gender | Female / Male / Diverse (about 1%) | **No: protected attribute** |
| DateofBirth | ISO date (YYYY-MM-DD). Employees were 20 to 60 (completed years) on their hire date; nobody is older than 66 at the cut-off | **No: protected attribute.** Derive age for reporting only |
| Salary | Latest annual salary in EUR | Yes |
| DateofHire | ISO date (YYYY-MM-DD) | Via tenure, not raw |
| DateofTermination | Exit date, **blank if still active** | **No: leakage** |
| EmploymentStatus | Active / Voluntary Resignation / Terminated for Cause | **No: this is the target** |
| TermReason | Reason for leaving, blank if active. Resignations: Better opportunity, Better salary, Dissatisfaction with management, Work hours / overtime, Career change, Relocation, Retirement. For cause: Performance, Attendance, Misconduct | **No: leakage** |
| PerformanceScore | Exceeds, Fully Meets, Needs Improvement, PIP | Yes |
| EngagementSurvey | Score from 1.5 to 5.0 | Yes |
| EmpSatisfaction | 1 to 5 | Yes |
| SpecialProjectsCount | Projects beyond the normal role | Yes |
| DaysLateLast30 | Days late in the last 30 days | Yes |
| Absences | Absence days | Yes |
| OvertimeHoursPerMonth | Average monthly overtime hours | Yes |
| YearsSinceLastPromotion | Years since last promotion (capped at tenure) | Yes |

## Rules for modelling

1. **Leakage.** `DateofTermination`, `EmploymentStatus` and `TermReason` record the outcome. Build your target from `EmploymentStatus`, then drop all three before training.
2. **Protected attributes.** Gender and DateofBirth (age) have **no effect on exits** in this data. Keep them for diversity reporting, not for prediction. If you test them, a good model should not need them.
3. **Tenure.** Compute it from `DateofHire` to the exit date (leavers) or to 2026-09-30 (active). Note that this uses the outcome date, so for a stricter model build snapshots (see the project plan).
4. **Snapshot columns.** Performance, engagement, satisfaction, overtime, lateness and absences are the latest recorded values (at exit for leavers).
5. **Small groups.** "Diverse" is 15 people. Percentages for groups this small swing wildly, so don't draw conclusions from them.

## Patterns planted in the data

These patterns are planted by the generator. A model that finds them shows the pipeline works, not that these drivers hold in real organisations.

| Pattern | Where to look |
|---|---|
| Hiring freeze | Jul-Dec 2023: hires drop sharply, headcount dips |
| Sales reorg | Sales exits roughly 3x normal Jan-Jun 2024, then a quiet stretch |
| IT/IS poaching wave | IT/IS exits spike Oct-Dec 2025 |
| Seasonality | Exits peak Feb-Mar, lowest in December |
| Early exits | Highest risk in the first 12 months |
| Leakiest department | Sales (about half of everyone ever employed there has left) |
| High-attrition managers | PRD-M4, SLS-M2, SWE-M3 |
| Source quality | Employee Referral keeps people longest; Indeed and Google Search lose the most |
| Individual drivers | Higher overtime, lower engagement, below-band pay and long gaps without promotion raise exit risk |
| Involuntary exits | Driven by PerformanceScore (PIP and Needs Improvement) and lateness/absence |

## Not included

There is no headcount plan or budget file. For variance analysis, create a quarterly plan as a documented assumption, or backtest your forecast against actual quarters.
