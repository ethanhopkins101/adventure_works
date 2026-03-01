# adventure_works
Full end to end ml pipeline that connects to Powerbi

# 🚲 AdventureWorks: Advanced Predictive Analytics & Strategy Suite

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Prophet%20%7C%20AutoARIMA-orange.svg)](https://facebook.github.io/prophet/)
[![ML](https://img.shields.io/badge/ML-Regression%20%26%20Clustering-green.svg)](https://scikit-learn.org/)

An enterprise-grade analytical ecosystem designed to transform the **AdventureWorks** retail dataset into a proactive decision-making engine. This project moves beyond static reporting into the realm of **Prescriptive Analytics**, offering optimized solutions for inventory, marketing, and customer retention.

---

## 📑 Table of Contents
1. [Data Engineering & ETL](#-data-engineering--etl)
2. [Sales Forecasting & Inventory](#-sales-forecasting--inventory)
3. [Returns & Risk Modeling](#-returns--risk-modeling)
4. [Customer Intelligence (CLV)](#-customer-intelligence-clv)
5. [Marketing Mix Modeling (MMM)](#-marketing-mix-modeling-mmm)
6. [Market Basket Analysis](#-market-basket-analysis)
7. [Price Elasticity & Promotions](#-price-elasticity--promotions)

---

## 🛠 Data Engineering & ETL
Before modeling, the raw SQL-style CSVs underwent a rigorous cleaning pipeline to ensure temporal and relational integrity.

* **Handling Orphaned Keys:** Detected and handled missing `ProductKey` references (specifically keys 480 & 529) that caused data leakage in standard joins.
* **Temporal Alignment:** Created a continuous daily grid to solve the "sparse data" problem, ensuring that days with zero sales were explicitly represented—a requirement for accurate time-series forecasting.
* **Normalization:** Implemented `StandardScaler` for clustering and `LabelEncoding` for categorical features used in regression.

---

## 📈 Sales Forecasting & Inventory
**Objective:** Predict daily demand for the next 30 days to optimize warehouse stocking levels.

### The Approach: Hybrid Time-Series
* **AutoARIMA:** Selected for high-volume subcategories (e.g., *Mountain Bikes*, *Road Bikes*). It automatically tunes $(p, d, q)$ parameters to capture seasonality and trends.
* **ColdStart Logic:** For new or low-volume products where historical data is insufficient, we used a volatility-adjusted mean approach to prevent over-stocking.
* **Stocking Logic:** * **Error-Adjusted:** Safety stock is calculated based on the Root Mean Squared Error (RMSE) of the forecast.
    * **Recommendation:** `Total Stock = Forecasted Volume + (1.96 * Std_Dev_Error)`.



---

## 🛡 Returns & Risk Modeling
**Objective:** Anticipate the volume of returns to protect net margins.

* **Algorithm:** **Facebook Prophet**.
* **Why Prophet?** It excels at handling "holidays" and "yearly seasonality" which are primary drivers for retail returns (e.g., post-Christmas surges).
* **Confidence Scoring:** We developed a custom Confidence Rating (%) based on the spread between the Upper and Lower forecast bounds. Items like *Tires and Tubes* maintain an **85% confidence**, whereas *Touring Bikes* use a conservative **65% rating**.

---

## 👥 Customer Intelligence (CLV)
**Objective:** Segment the customer base and predict their 90-day future value.

* **Algorithms:** **BG/NBD** (BetaGeo/Negative Binomial Distribution) for purchase frequency and **Gamma-Gamma** for monetary value.
* **Segmentation Logic:** Used **K-Means Clustering** on RFM (Recency, Frequency, Monetary) metrics.
* **Actionable Segments:**
    | Segment | Description | Strategy |
    | :--- | :--- | :--- |
    | **Elite Champions** | High R, High F, High M | Retention & Early Access |
    | **Slipping Spenders** | Low R, High M | Urgent "Win-Back" campaigns |
    | **Growth Potential** | High R, Low F | Conversion to habitual buyers |

---

## 💰 Marketing Mix Modeling (MMM)
**Objective:** Determine the optimal ad spend distribution to maximize Profit.

* **Approach:** **Adstock Transformation** & **OLS Regression**.
* **Logic:** We modeled the "carry-over" effect of advertising (Adstock) to recognize that a TV ad seen today influences a purchase 2 weeks from now.
* **ROI Highlights:** * **Facebook:** Highest ROI (1.77) — Recommended for scaling.
    * **Search/Print:** Lower direct ROI — Recommended for baseline visibility only.

---

## 🛒 Market Basket Analysis
**Objective:** Identify product affinities to drive cross-selling.

* **Algorithm:** **Apriori** & **Association Rules**.
* **Key Insight:** *Hydration Packs* and *Bottles and Cages* show a **Lift of 2.63**, meaning a customer is 2.6x more likely to buy both together than at random.
* **Implementation:** These rules inform the "Customers also bought" section of the e-commerce engine.

---

## 🏷 Price Elasticity & Promotions
**Objective:** Predict how price changes affect total profit during major events.

* **Algorithm:** **Generalized Additive Models (GAMs)**.
* **Scenario Modeling:** Simulates performance across "Black Friday," "Cyber Monday," and "Back to School."
* **Results:** Identified that for *Mountain-100 Black*, a price drop during Black Friday increases volume by 4.4x, but requires careful margin monitoring to remain profitable.



---

## 🚀 How to Use
1.  **Exploration:** Start with `notebooks/data_cleaning.ipynb` to see the ETL process.
2.  **Simulation:** Use `reports/budget_simulations.json` to view pre-calculated marketing outcomes.
3.  **Deployment:** Load `models/encoder.json` to transform new production data for the forecasting models.

---
*Developed with a commitment to scientific objectivity and simple, actionable insights.*