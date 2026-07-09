"""
=====================================================================
SUPPLY CHAIN INVENTORY OPTIMIZATION ANALYTICS
Module 1: Data Preprocessing & Feature Engineering
=====================================================================
Author : Data Analytics Portfolio Project
Purpose: Generate a realistic supply chain dataset, clean it,
         engineer business-relevant features, and save a
         production-ready dataset for downstream EDA and ML.
=====================================================================
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "raw_supply_chain_data.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_supply_chain_data.csv")


# =====================================================================
# STEP 1: SYNTHETIC (REALISTIC) DATASET GENERATION
# =====================================================================
def generate_dataset(n_records=6000):
    """
    Generates a realistic synthetic supply chain dataset.
    In a real-world project, this would be replaced by an
    actual ERP / WMS / POS data extract (SQL, CSV, API).
    """

    print("Generating synthetic supply chain dataset...")

    categories = ["Electronics", "Apparel", "Grocery", "Furniture",
                  "Automotive", "Pharmaceuticals", "Toys", "Cosmetics"]

    warehouses = ["Warehouse-North", "Warehouse-South",
                  "Warehouse-East", "Warehouse-West", "Warehouse-Central"]

    suppliers = ["Supplier-A", "Supplier-B", "Supplier-C",
                 "Supplier-D", "Supplier-E", "Supplier-F"]

    # Date range: 2 years of order data
    date_range = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

    n_products = 250
    product_ids = [f"PROD-{str(i).zfill(4)}" for i in range(1, n_products + 1)]

    # Assign fixed category/supplier per product for realism
    product_category_map = {p: np.random.choice(categories) for p in product_ids}
    product_supplier_map = {p: np.random.choice(suppliers) for p in product_ids}

    records = []
    for _ in range(n_records):
        order_date = np.random.choice(date_range)
        product_id = np.random.choice(product_ids)
        category = product_category_map[product_id]
        warehouse = np.random.choice(warehouses)
        supplier = product_supplier_map[product_id]

        stock_quantity = np.random.randint(0, 1000)
        units_sold = np.random.randint(0, max(1, int(stock_quantity * 0.8)) + 1)
        demand_forecast = units_sold + np.random.randint(-30, 60)
        demand_forecast = max(demand_forecast, 0)

        lead_time = np.random.randint(1, 20)          # days supplier takes
        shipping_time = np.random.randint(1, 10)       # days to ship to warehouse

        unit_cost = np.round(np.random.uniform(5, 500), 2)
        inventory_cost = np.round(stock_quantity * unit_cost * np.random.uniform(0.05, 0.15), 2)

        unit_price = np.round(unit_cost * np.random.uniform(1.2, 2.5), 2)
        revenue = np.round(units_sold * unit_price, 2)

        customer_demand = units_sold + np.random.randint(-20, 50)
        customer_demand = max(customer_demand, 0)

        records.append([
            order_date, product_id, category, warehouse, supplier,
            stock_quantity, units_sold, demand_forecast, lead_time,
            shipping_time, inventory_cost, revenue, customer_demand,
            unit_cost, unit_price
        ])

    df = pd.DataFrame(records, columns=[
        "Order Date", "Product ID", "Category", "Warehouse", "Supplier",
        "Stock Quantity", "Units Sold", "Demand Forecast", "Lead Time",
        "Shipping Time", "Inventory Cost", "Revenue", "Customer Demand",
        "Unit Cost", "Unit Price"
    ])

    # -----------------------------------------------------------
    # Inject realistic data-quality issues (so cleaning is meaningful)
    # -----------------------------------------------------------
    # 1. Missing values
    for col in ["Stock Quantity", "Units Sold", "Demand Forecast",
                "Lead Time", "Shipping Time", "Inventory Cost",
                "Revenue", "Customer Demand"]:
        missing_idx = df.sample(frac=0.02, random_state=np.random.randint(1000)).index
        df.loc[missing_idx, col] = np.nan

    # 2. Duplicate rows
    dup_rows = df.sample(frac=0.015, random_state=7)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3. Outliers (extreme values injected intentionally)
    outlier_idx = df.sample(frac=0.005, random_state=11).index
    df.loc[outlier_idx, "Inventory Cost"] = df.loc[outlier_idx, "Inventory Cost"] * 50
    df.loc[outlier_idx, "Revenue"] = df.loc[outlier_idx, "Revenue"] * 40

    # 4. Inconsistent data types (strings mixed into numeric columns)
    df["Stock Quantity"] = df["Stock Quantity"].astype(object)
    dtype_issue_idx = df.sample(frac=0.005, random_state=13).index
    df.loc[dtype_issue_idx, "Stock Quantity"] = "unknown"

    print(f"Raw dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =====================================================================
# STEP 2: DATA CLEANING
# =====================================================================
def clean_data(df):
    """
    Full data cleaning pipeline:
    - Data type correction
    - Missing value handling
    - Duplicate removal
    - Outlier detection & treatment (IQR method)
    """

    print("\n--- CLEANING DATA ---")
    df = df.copy()

    # -----------------------------------------------------------
    # 2.1 Fix data types
    # -----------------------------------------------------------
    numeric_cols = ["Stock Quantity", "Units Sold", "Demand Forecast",
                     "Lead Time", "Shipping Time", "Inventory Cost",
                     "Revenue", "Customer Demand", "Unit Cost", "Unit Price"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # invalid strings -> NaN

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    for col in ["Product ID", "Category", "Warehouse", "Supplier"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"Data types converted for {len(numeric_cols)} numeric columns.")

    # -----------------------------------------------------------
    # 2.2 Handle missing values
    # -----------------------------------------------------------
    missing_before = df.isnull().sum().sum()

    # Drop rows with missing critical identifiers
    df = df.dropna(subset=["Order Date", "Product ID", "Category"])

    # Numeric columns: impute with median grouped by Category (business-realistic)
    for col in numeric_cols:
        df[col] = df.groupby("Category")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())  # fallback for any remaining NaNs

    missing_after = df.isnull().sum().sum()
    print(f"Missing values handled: {missing_before} -> {missing_after}")

    # -----------------------------------------------------------
    # 2.3 Remove duplicates
    # -----------------------------------------------------------
    before_dup = df.shape[0]
    df = df.drop_duplicates()
    after_dup = df.shape[0]
    print(f"Duplicates removed: {before_dup - after_dup} rows")

    # -----------------------------------------------------------
    # 2.4 Outlier detection & treatment (IQR capping)
    # -----------------------------------------------------------
    outlier_cols = ["Inventory Cost", "Revenue", "Stock Quantity", "Units Sold"]
    total_outliers = 0

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        total_outliers += outliers.shape[0]

        # Cap outliers rather than deleting them (preserves data volume)
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    print(f"Outliers detected & capped (IQR method): {total_outliers} values across {len(outlier_cols)} columns")

    # -----------------------------------------------------------
    # 2.5 Logical consistency fixes
    # -----------------------------------------------------------
    df["Stock Quantity"] = df["Stock Quantity"].clip(lower=0)
    df["Units Sold"] = df["Units Sold"].clip(lower=0)
    df["Demand Forecast"] = df["Demand Forecast"].clip(lower=0)
    df["Customer Demand"] = df["Customer Demand"].clip(lower=0)
    df["Lead Time"] = df["Lead Time"].clip(lower=1)
    df["Shipping Time"] = df["Shipping Time"].clip(lower=1)

    df = df.reset_index(drop=True)
    print(f"Final cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


# =====================================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================================
def engineer_features(df):
    """
    Creates business-critical KPIs and features used across
    EDA dashboards and ML models.
    """

    print("\n--- FEATURE ENGINEERING ---")
    df = df.copy()

    # -----------------------------------------------------------
    # 1. Inventory Turnover Ratio = Units Sold / Avg Stock Quantity
    #    (higher = faster moving inventory)
    # -----------------------------------------------------------
    df["Inventory Turnover Ratio"] = np.round(
        df["Units Sold"] / df["Stock Quantity"].replace(0, np.nan), 3
    )
    df["Inventory Turnover Ratio"] = df["Inventory Turnover Ratio"].fillna(0)

    # -----------------------------------------------------------
    # 2. Stock Availability % = (Stock Quantity / (Stock + Units Sold)) * 100
    # -----------------------------------------------------------
    denom = (df["Stock Quantity"] + df["Units Sold"]).replace(0, np.nan)
    df["Stock Availability %"] = np.round((df["Stock Quantity"] / denom) * 100, 2)
    df["Stock Availability %"] = df["Stock Availability %"].fillna(0)

    # -----------------------------------------------------------
    # 3. Reorder Flag: 1 if stock is below forecasted demand threshold
    # -----------------------------------------------------------
    df["Reorder Flag"] = np.where(
        df["Stock Quantity"] < (df["Demand Forecast"] * 0.5), 1, 0
    )

    # -----------------------------------------------------------
    # 4. Demand Category: classify products by sales velocity
    # -----------------------------------------------------------
    def classify_demand(row):
        if row["Units Sold"] >= 400:
            return "High Demand"
        elif row["Units Sold"] >= 150:
            return "Medium Demand"
        else:
            return "Low Demand"

    df["Demand Category"] = df.apply(classify_demand, axis=1)

    # -----------------------------------------------------------
    # 5. Profit Calculation = Revenue - Inventory Cost
    # -----------------------------------------------------------
    df["Profit"] = np.round(df["Revenue"] - df["Inventory Cost"], 2)
    df["Profit Margin %"] = np.round(
        (df["Profit"] / df["Revenue"].replace(0, np.nan)) * 100, 2
    )
    df["Profit Margin %"] = df["Profit Margin %"].fillna(0)

    # -----------------------------------------------------------
    # 6. Supplier Delay Indicator: 1 if Lead Time exceeds acceptable threshold
    # -----------------------------------------------------------
    lead_time_threshold = df["Lead Time"].quantile(0.75)
    df["Supplier Delay Indicator"] = np.where(
        df["Lead Time"] > lead_time_threshold, 1, 0
    )

    # -----------------------------------------------------------
    # 7. Additional useful business fields
    # -----------------------------------------------------------
    df["Total Fulfillment Time"] = df["Lead Time"] + df["Shipping Time"]
    df["Stock Risk"] = np.where(
        df["Stock Quantity"] < df["Demand Forecast"], "Understocked",
        np.where(df["Stock Quantity"] > df["Demand Forecast"] * 2, "Overstocked", "Balanced")
    )

    df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Order Year"] = df["Order Date"].dt.year

    print("Engineered features: Inventory Turnover Ratio, Stock Availability %, "
          "Reorder Flag, Demand Category, Profit, Profit Margin %, "
          "Supplier Delay Indicator, Total Fulfillment Time, Stock Risk")

    return df


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("SUPPLY CHAIN INVENTORY OPTIMIZATION - DATA PREPROCESSING")
    print("=" * 70)

    # Step 1: Generate / Load raw dataset
    raw_df = generate_dataset(n_records=6000)
    raw_df.to_csv(RAW_PATH, index=False)
    print(f"\nRaw dataset saved to: {RAW_PATH}")

    # Step 2: Clean dataset
    cleaned_df = clean_data(raw_df)

    # Step 3: Feature engineering
    final_df = engineer_features(cleaned_df)

    # Step 4: Save cleaned + engineered dataset
    final_df.to_csv(CLEANED_PATH, index=False)
    print(f"\nCleaned & feature-engineered dataset saved to: {CLEANED_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Final Records        : {final_df.shape[0]}")
    print(f"Final Columns        : {final_df.shape[1]}")
    print(f"Date Range           : {final_df['Order Date'].min().date()} to {final_df['Order Date'].max().date()}")
    print(f"Unique Products      : {final_df['Product ID'].nunique()}")
    print(f"Unique Categories    : {final_df['Category'].nunique()}")
    print(f"Unique Warehouses    : {final_df['Warehouse'].nunique()}")
    print(f"Unique Suppliers     : {final_df['Supplier'].nunique()}")
    print("=" * 70)
    print("Preprocessing complete. Ready for EDA (eda_analysis.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()
