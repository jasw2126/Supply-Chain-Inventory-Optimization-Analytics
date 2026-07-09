# 📦 Supply Chain Inventory Optimization Analytics

**A complete end-to-end Data Analytics portfolio project** covering data cleaning, exploratory data analysis, business intelligence dashboards, and machine learning — built entirely in Python.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?logo=pandas)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project simulates a real-world **Supply Chain & Inventory Management analytics pipeline** for a multi-category retail/distribution business operating across multiple warehouses and suppliers.

It walks through the full analytics lifecycle a Data Analyst / BI Analyst performs in industry:

1. **Data Preprocessing** — cleaning messy, real-world-style transactional data
2. **Exploratory Data Analysis (EDA)** — inventory, sales, supplier, and warehouse analytics
3. **Business Intelligence Dashboards** — professional PNG report generation
4. **Machine Learning** — demand forecasting and inventory risk prediction

The result is a decision-support toolkit that helps stakeholders answer questions like:
*Which products are we about to run out of? Which suppliers are slowing us down? Where is our profit actually coming from? Can we predict demand and stock-out risk before it happens?*

---

## 🎯 Business Problem

Retail and distribution companies routinely lose revenue due to:

- **Stockouts** — high-demand products running out, resulting in lost sales
- **Overstocking** — excess capital tied up in slow-moving inventory
- **Unreliable suppliers** — delayed lead times disrupting fulfillment
- **Poor visibility** — no centralized view of inventory health across warehouses

This project builds an analytics solution that identifies these problems from historical operations data and uses machine learning to **proactively flag at-risk inventory** and **forecast future demand**, enabling smarter purchasing and replenishment decisions.

---

## 🛠️ Tech Stack

| Category            | Tools / Libraries                          |
|---------------------|---------------------------------------------|
| Language             | Python 3.12                                |
| Data Manipulation    | Pandas, NumPy                              |
| Data Visualization   | Matplotlib, Seaborn                        |
| Machine Learning      | Scikit-learn (Random Forest Regressor & Classifier) |
| Model Persistence     | Joblib                                    |
| Report Output         | PNG dashboards (static, shareable BI reports) |

> ⚠️ This is a **pure Python analytics project** — no Streamlit, no web frameworks, no dashboards requiring a live server. All deliverables are scripts and static PNG reports, exactly as used in real BI reporting workflows (e.g., emailed reports, Power BI/Tableau export style visuals).

---

## 🗃️ Dataset Details

The dataset is a **synthetically generated but realistic** supply chain transactional dataset (`data_preprocessing.py` generates it programmatically, mimicking an ERP/WMS data extract). This design choice keeps the project fully reproducible and self-contained.

**Scale:** ~6,000 transactional records | 2 years (2023–2024) | 250 unique products | 8 categories | 5 warehouses | 6 suppliers

### Raw Columns

| Column | Description |
|---|---|
| Order Date | Date of the transaction |
| Product ID | Unique product identifier |
| Category | Product category (Electronics, Apparel, Grocery, etc.) |
| Warehouse | Warehouse fulfilling the order |
| Supplier | Supplier responsible for restocking the product |
| Stock Quantity | Units currently in stock |
| Units Sold | Units sold in the transaction period |
| Demand Forecast | Forecasted demand for the product |
| Lead Time | Days the supplier takes to deliver stock |
| Shipping Time | Days to ship from warehouse to customer |
| Inventory Cost | Cost of holding current inventory |
| Revenue | Revenue generated from units sold |
| Customer Demand | Actual customer demand signal |

### Engineered Features (via `data_preprocessing.py`)

| Feature | Business Meaning |
|---|---|
| Inventory Turnover Ratio | Units Sold ÷ Stock Quantity — how fast inventory moves |
| Stock Availability % | Share of demand that could be met from available stock |
| Reorder Flag | Flags products that need replenishment |
| Demand Category | High / Medium / Low demand classification |
| Profit & Profit Margin % | Revenue minus Inventory Cost, and margin % |
| Supplier Delay Indicator | Flags suppliers exceeding the 75th percentile lead time |
| Total Fulfillment Time | Lead Time + Shipping Time |
| Stock Risk | Understocked / Balanced / Overstocked classification |

The raw data intentionally includes **missing values, duplicate rows, inconsistent data types, and outliers** so the cleaning pipeline reflects real-world data quality challenges.

---

## 🔍 Analysis Performed

### 📦 Inventory Analysis
- Highest-selling products
- Slow-moving / stagnant products
- Stock shortage (understocked) products
- Overstocked products tying up capital

### 💰 Sales Analytics
- Revenue by category
- Monthly sales trend (2023–2024)
- Top products by revenue

### 🚚 Supplier Analytics
- Supplier delay rate comparison
- Average lead time by supplier
- Total fulfillment time benchmarking

### 🏬 Warehouse Analytics
- Warehouse efficiency (inventory turnover)
- Stock distribution across warehouses

---

## 📊 Dashboard Images

All dashboards are generated by `eda_analysis.py` and `ml_model.py`, saved as high-resolution PNGs in the `visuals/` folder — ready to drop into a resume portfolio, PowerPoint, or GitHub README.

### 1️⃣ Executive Dashboard
Company-wide KPIs: Total Revenue, Total Products, Inventory Cost, Stock Risk Count, plus revenue trends, category profit, warehouse & supplier snapshots.

`visuals/executive_dashboard.png`

### 2️⃣ Inventory Dashboard
Stock level distribution, fast vs. slow-moving products, and reorder alerts by category.

`visuals/inventory_dashboard.png`

### 3️⃣ Supplier Dashboard
Supplier delay rates, lead time comparison, lead time distribution, and fulfillment time benchmarking.

`visuals/supplier_dashboard.png`

### 4️⃣ Sales Dashboard
Monthly revenue trend, category-wise sales, and top 10 products by revenue.

`visuals/sales_dashboard.png`

### 5️⃣ ML Feature Importance Report
Feature importance rankings for both the demand forecasting and inventory risk models.

`visuals/ml_feature_importance.png`

### 6️⃣ ML Prediction Comparison Report
Model KPIs, actual vs. predicted demand scatter plot, residual error distribution, and inventory risk confusion matrix.

`visuals/ml_prediction_comparison.png`

> 💡 Open the PNG files in the `visuals/` folder to view all six dashboards.

---

## 🤖 Machine Learning Results

Two Random Forest models were built with `ml_model.py`, trained on an 80/20 split, with categorical features label-encoded and leakage-prone columns (e.g., Revenue, Stock Availability %) intentionally excluded from the demand model for realistic generalization performance.

### 1. Demand Forecasting — Random Forest Regressor
Predicts realized product demand (Units Sold) from operational features (stock levels, lead/shipping time, cost structure, category, warehouse, supplier).

| Metric | Score |
|---|---|
| MAE | ~104.5 |
| RMSE | ~138.1 |
| R² Score | ~0.39 |

### 2. Inventory Risk Classification — Random Forest Classifier
Predicts whether a product is at **Low Stock Risk** (understocked) or **Safe Stock**, using stock levels, demand forecast, turnover ratio, and supplier delay signals. Class-balanced to handle the natural rarity of stockout events.

| Metric | Score |
|---|---|
| Accuracy | ~98.2% |
| Precision | ~80.8% |
| Recall | ~77.8% |
| F1 Score | ~79.3% |

Both models, their label encoders, and feature lists are saved to the `models/` folder using `joblib` for reuse in production or further evaluation.

---

## 📁 Folder Structure

```
Supply_Chain_Inventory_Analytics/
├── data/
│   ├── raw_supply_chain_data.csv          # Generated raw dataset (with quality issues)
│   └── cleaned_supply_chain_data.csv      # Cleaned + feature-engineered dataset
├── visuals/
│   ├── executive_dashboard.png
│   ├── inventory_dashboard.png
│   ├── supplier_dashboard.png
│   ├── sales_dashboard.png
│   ├── ml_feature_importance.png
│   └── ml_prediction_comparison.png
├── models/
│   ├── demand_forecast_model.pkl
│   ├── demand_forecast_encoders.pkl
│   ├── demand_forecast_features.pkl
│   ├── inventory_risk_model.pkl
│   ├── inventory_risk_encoders.pkl
│   └── inventory_risk_features.pkl
├── data_preprocessing.py                  # Data generation, cleaning, feature engineering
├── eda_analysis.py                        # EDA, business analytics, dashboard PNG generation
├── ml_model.py                            # ML models, evaluation, ML report PNG generation
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate, clean, and feature-engineer the dataset
python data_preprocessing.py

# 3. Run EDA and generate BI dashboards
python eda_analysis.py

# 4. Train ML models and generate ML reports
python ml_model.py
```

All outputs (cleaned data, dashboards, trained models) are generated automatically into the `data/`, `visuals/`, and `models/` folders.

---

## 📝 Resume Description

> **Supply Chain Inventory Optimization Analytics** — Built an end-to-end Python analytics pipeline analyzing 6,000+ supply chain transactions across 250 products, 5 warehouses, and 6 suppliers; engineered 9 business KPIs (inventory turnover, stock risk, profit margin, supplier delay), produced 6 professional BI dashboard reports (Matplotlib/Seaborn), and developed Random Forest models for demand forecasting (R² 0.39) and inventory risk classification (98% accuracy, 79% F1) using Scikit-learn, enabling data-driven reorder and supplier-performance decisions.

---

## 👤 Author

Data Analytics Portfolio Project — built to demonstrate practical, production-style skills in data cleaning, business analytics, dashboard reporting, and applied machine learning using only core Python data science libraries.

---

## 📄 License

This project is released under the MIT License — free to use, modify, and share for learning and portfolio purposes.
