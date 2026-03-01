# adventure_works
Full end to end ml pipeline that connects to Powerbi

# 🚴 Adventure Works: Predictive Business Intelligence Ecosystem

Welcome to the **Adventure Works Predictive Suite**. This project transforms raw transactional data into a sophisticated decision-making engine. By blending Bayesian inference, time-series forecasting, and non-linear regression, we provide a 360-degree view of future business performance.

---

## 📊 Strategic Study Overview

This project was designed to solve five core business challenges:
1.  **Demand Sensing:** When will our inventory peak?
2.  **Revenue Protection:** How many returns should we expect next month?
3.  **Capital Allocation:** Which marketing channels yield the highest Profit ROI?
4.  **Customer Equity:** Who are our most valuable "Whale" customers?
5.  **Pricing Strategy:** What is the optimal price point for a "Cyber Monday" vs. "Easter" promo?

---

## 🔬 Methodology & Model Selection

We prioritized **probabilistic models** over simple averages to capture the inherent uncertainty in retail.

### 1. Sales & Returns Forecasting 📈
We evaluated traditional ARIMA models but ultimately selected **Facebook Prophet** for its ability to handle "Multi-Period Seasonality" (e.g., the massive summer bike peak vs. holiday accessories).

* **Sales Forecast:** Adjusted to predict the 2017 growth trajectory.
* **Returns Forecast:** We moved away from static return rates. Instead, we used a **Lagged Sales Heuristic**, acknowledging that a return today is a function of a sale made 15–30 days ago.
* **Scenario Adjustment:** We modeled "High-Growth" vs. "Baseline" scenarios to help the warehouse team prepare for +/- 15% fluctuations in return volume.

### 2. Marketing Mix Modeling (MMM) 📣
Using **Orbit (Bayesian DLT)**, we calculated the "Adstock Effect"—the memory decay of an advertisement.

| Channel | Decay (Alpha) | Logic |
| :--- | :--- | :--- |
| **TV** | 0.70 | High "brand equity" memory; long-term impact. |
| **Print** | 0.60 | Physical shelf-life; slow decay. |
| **Facebook** | 0.30 | High immediate engagement; fast decay. |
| **Search** | 0.20 | Direct intent; zero "memory" post-click. |

### 3. Price Elasticity & Promo Optimization 🏷️
Standard linear models failed to capture the "cliff" where price becomes too high for consumers. We implemented **Generalized Additive Models (GAMs)** to find the profit-maximizing "sweet spot."

* **Scenarios Handled:** No Promo, Black Friday, Cyber Monday, and Back to School.
* **Insight:** We found that during "Cyber Monday," volume is so elastic that a 20% price cut yields a **3x increase in net profit** despite lower margins.

---

## 🛠️ Technical Stack

This project leverages state-of-the-art Python libraries:

* **Forecasting:** [Prophet](https://facebook.github.io/prophet/) — Robust time-series.
* **Bayesian MMM:** [Orbit-ML](https://orbit-ml.readthedocs.io/) — Object-oriented Bayesian modeling.
* **Customer Value:** [Lifetimes](https://lifetimes.readthedocs.io/) — BG/NBD & Gamma-Gamma models.
* **Optimization:** [PyGAM](https://pygam.readthedocs.io/) — Flexible non-linear regression.
* **Data Ops:** `Pandas`, `Scikit-Learn`, `Joblib`.

---

## 📁 Project Structure

```text
├── data/
│   ├── cleaned/             # Source CSVs (Sales, Returns, Customers)
│   └── models/              # Final processed output tables
├── models/
│   ├── association_rule/    # Market Basket Analysis (Apriori)
│   ├── bayesian_mmm/        # Marketing ROI models
│   ├── price_elasticity/    # GAM optimization models
│   └── clv/                 # K-Means & Probabilistic CLV
├── pictures/                # Optimization curves & waterfall charts
└── src/                     # Modular execution scripts (.py)