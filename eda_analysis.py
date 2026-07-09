"""
=====================================================================
SUPPLY CHAIN INVENTORY OPTIMIZATION ANALYTICS
Module 2: Exploratory Data Analysis & Business Intelligence Dashboards
=====================================================================
Author : Data Analytics Portfolio Project
Purpose: Perform inventory, sales, supplier, and warehouse analytics
         and generate professional PNG dashboards (BI-style reports).
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_supply_chain_data.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.titleweight"] = "bold"

COLOR_PRIMARY = "#1F4E79"     # deep blue
COLOR_SECONDARY = "#2E86AB"   # medium blue
COLOR_ACCENT = "#F39C12"      # orange
COLOR_DANGER = "#E74C3C"      # red
COLOR_SUCCESS = "#27AE60"     # green
COLOR_GREY = "#7F8C8D"
PALETTE = ["#1F4E79", "#2E86AB", "#F39C12", "#27AE60", "#E74C3C",
           "#8E44AD", "#16A085", "#D35400"]


def load_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def kpi_card(ax, title, value, subtitle="", color=COLOR_PRIMARY):
    """Draws a professional KPI card inside a given axis."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                          facecolor=color, edgecolor="none", zorder=1,
                          alpha=0.95)
    ax.add_patch(card)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22,
            fontweight="bold", color="white", transform=ax.transAxes, zorder=2)
    ax.text(0.5, 0.30, title, ha="center", va="center", fontsize=11,
            color="white", transform=ax.transAxes, zorder=2)
    if subtitle:
        ax.text(0.5, 0.12, subtitle, ha="center", va="center", fontsize=9,
                color="#EAEAEA", transform=ax.transAxes, zorder=2)


def add_dashboard_title(fig, title, subtitle):
    fig.suptitle(title, fontsize=20, fontweight="bold", color=COLOR_PRIMARY,
                 x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.955, subtitle, fontsize=11, color=COLOR_GREY, ha="left")


# =====================================================================
# ANALYTICS FUNCTIONS (printed business insights)
# =====================================================================
def inventory_analysis(df):
    print("=" * 70)
    print("INVENTORY ANALYSIS")
    print("=" * 70)

    top_selling = df.groupby("Product ID")["Units Sold"].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Highest Selling Products:\n", top_selling)

    slow_moving = df.groupby("Product ID")["Units Sold"].sum().sort_values().head(10)
    print("\nTop 10 Slow Moving Products:\n", slow_moving)

    shortage = df[df["Stock Risk"] == "Understocked"]["Product ID"].value_counts().head(10)
    print("\nTop 10 Stock Shortage Products:\n", shortage)

    overstock = df[df["Stock Risk"] == "Overstocked"]["Product ID"].value_counts().head(10)
    print("\nTop 10 Overstock Products:\n", overstock)

    return top_selling, slow_moving, shortage, overstock


def sales_analytics(df):
    print("\n" + "=" * 70)
    print("SALES ANALYTICS")
    print("=" * 70)

    revenue_by_category = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    print("\nRevenue by Category:\n", revenue_by_category)

    monthly_sales = df.groupby("Order Month")["Revenue"].sum().sort_index()
    print("\nMonthly Sales Trend (first 5 months):\n", monthly_sales.head())

    product_performance = df.groupby("Product ID")["Revenue"].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Products by Revenue:\n", product_performance)

    return revenue_by_category, monthly_sales, product_performance


def supplier_analytics(df):
    print("\n" + "=" * 70)
    print("SUPPLIER ANALYTICS")
    print("=" * 70)

    supplier_delay = df.groupby("Supplier")["Supplier Delay Indicator"].mean().sort_values(ascending=False) * 100
    print("\nSupplier Delay % (of orders delayed):\n", supplier_delay)

    lead_time_comp = df.groupby("Supplier")["Lead Time"].mean().sort_values(ascending=False)
    print("\nAverage Lead Time by Supplier (days):\n", lead_time_comp)

    return supplier_delay, lead_time_comp


def warehouse_analytics(df):
    print("\n" + "=" * 70)
    print("WAREHOUSE ANALYTICS")
    print("=" * 70)

    warehouse_efficiency = df.groupby("Warehouse")["Inventory Turnover Ratio"].mean().sort_values(ascending=False)
    print("\nWarehouse Efficiency (Avg Turnover Ratio):\n", warehouse_efficiency)

    stock_distribution = df.groupby("Warehouse")["Stock Quantity"].sum().sort_values(ascending=False)
    print("\nStock Distribution by Warehouse:\n", stock_distribution)

    return warehouse_efficiency, stock_distribution


# =====================================================================
# DASHBOARD 1: EXECUTIVE DASHBOARD
# =====================================================================
def build_executive_dashboard(df):
    print("\nBuilding executive_dashboard.png ...")

    total_revenue = df["Revenue"].sum()
    total_products = df["Product ID"].nunique()
    total_inventory_cost = df["Inventory Cost"].sum()
    stock_risk_count = df[df["Stock Risk"].isin(["Understocked", "Overstocked"])].shape[0]
    total_profit = df["Profit"].sum()
    avg_margin = df["Profit Margin %"].mean()

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[0.6, 1.3, 1.3, 1.3],
                            hspace=0.65, wspace=0.35, top=0.90, bottom=0.06, left=0.06, right=0.97)

    add_dashboard_title(fig, "Executive Supply Chain Dashboard",
                         "Company-wide KPIs | Inventory, Revenue & Risk Overview")

    # KPI Cards Row
    kpi_card(fig.add_subplot(gs[0, 0]), "TOTAL REVENUE", f"${total_revenue/1e6:.2f}M", "All-time sales", COLOR_PRIMARY)
    kpi_card(fig.add_subplot(gs[0, 1]), "TOTAL PRODUCTS", f"{total_products:,}", "Active SKUs", COLOR_SECONDARY)
    kpi_card(fig.add_subplot(gs[0, 2]), "INVENTORY COST", f"${total_inventory_cost/1e6:.2f}M", "Holding cost", COLOR_ACCENT)
    kpi_card(fig.add_subplot(gs[0, 3]), "STOCK RISK COUNT", f"{stock_risk_count:,}", "Under/Overstocked records", COLOR_DANGER)

    # Row 2: Revenue trend + Profit margin distribution
    ax1 = fig.add_subplot(gs[1, :3])
    monthly = df.groupby("Order Month")["Revenue"].sum().sort_index()
    ax1.plot(monthly.index, monthly.values, marker="o", color=COLOR_PRIMARY, linewidth=2)
    ax1.fill_between(range(len(monthly)), monthly.values, color=COLOR_PRIMARY, alpha=0.1)
    ax1.set_title("Monthly Revenue Trend", fontsize=13, color=COLOR_PRIMARY)
    ax1.tick_params(axis="x", rotation=60, labelsize=8)
    ax1.set_ylabel("Revenue ($)")

    ax2 = fig.add_subplot(gs[1, 3])
    risk_counts = df["Stock Risk"].value_counts()
    colors_map = {"Balanced": COLOR_SUCCESS, "Understocked": COLOR_DANGER, "Overstocked": COLOR_ACCENT}
    ax2.pie(risk_counts.values, labels=risk_counts.index, autopct="%1.0f%%", startangle=90,
            colors=[colors_map.get(i, COLOR_GREY) for i in risk_counts.index],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 8})
    ax2.set_title("Stock Risk Breakdown", fontsize=12, color=COLOR_PRIMARY)

    # Row 3: Revenue by category + Profit by category
    ax3 = fig.add_subplot(gs[2, :2])
    rev_cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    sns.barplot(x=rev_cat.values, y=rev_cat.index, ax=ax3, palette=PALETTE)
    ax3.set_title("Revenue by Category", fontsize=13, color=COLOR_PRIMARY)
    ax3.set_xlabel("Revenue ($)")
    ax3.set_ylabel("")

    ax4 = fig.add_subplot(gs[2, 2:])
    profit_cat = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
    bars = ax4.bar(profit_cat.index, profit_cat.values, color=COLOR_SECONDARY)
    ax4.set_title("Profit by Category", fontsize=13, color=COLOR_PRIMARY)
    ax4.tick_params(axis="x", rotation=45, labelsize=8)
    ax4.set_ylabel("Profit ($)")

    # Row 4: Warehouse stock distribution + Supplier delay snapshot
    ax5 = fig.add_subplot(gs[3, :2])
    wh_stock = df.groupby("Warehouse")["Stock Quantity"].sum().sort_values(ascending=False)
    sns.barplot(x=wh_stock.index, y=wh_stock.values, ax=ax5, palette=PALETTE)
    ax5.set_title("Stock Distribution by Warehouse", fontsize=13, color=COLOR_PRIMARY)
    ax5.tick_params(axis="x", rotation=20, labelsize=8)
    ax5.set_ylabel("Stock Qty")

    ax6 = fig.add_subplot(gs[3, 2:])
    sup_delay = df.groupby("Supplier")["Supplier Delay Indicator"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=sup_delay.index, y=sup_delay.values, ax=ax6, palette=PALETTE)
    ax6.set_title("Supplier Delay % Snapshot", fontsize=13, color=COLOR_PRIMARY)
    ax6.tick_params(axis="x", rotation=20, labelsize=8)
    ax6.set_ylabel("Delay %")

    fig.text(0.02, 0.01, f"Total Profit: ${total_profit/1e6:.2f}M   |   Avg Profit Margin: {avg_margin:.1f}%   |   Generated by Supply Chain Analytics Pipeline",
              fontsize=9, color=COLOR_GREY)

    out_path = os.path.join(VISUALS_DIR, "executive_dashboard.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# DASHBOARD 2: INVENTORY DASHBOARD
# =====================================================================
def build_inventory_dashboard(df):
    print("Building inventory_dashboard.png ...")

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.55, 1.3, 1.3],
                            hspace=0.65, wspace=0.4, top=0.90, bottom=0.06, left=0.06, right=0.97)

    add_dashboard_title(fig, "Inventory Management Dashboard",
                         "Stock Levels | Fast vs Slow Movers | Reorder Alerts")

    reorder_count = df[df["Reorder Flag"] == 1].shape[0]
    understock_count = df[df["Stock Risk"] == "Understocked"].shape[0]
    overstock_count = df[df["Stock Risk"] == "Overstocked"].shape[0]
    avg_turnover = df["Inventory Turnover Ratio"].mean()

    kpi_card(fig.add_subplot(gs[0, 0]), "REORDER ALERTS", f"{reorder_count:,}", "Products to reorder", COLOR_DANGER)
    kpi_card(fig.add_subplot(gs[0, 1]), "UNDERSTOCKED", f"{understock_count:,}", "Risk of stockout", COLOR_ACCENT)
    kpi_card(fig.add_subplot(gs[0, 2]), "OVERSTOCKED", f"{overstock_count:,}", "Excess inventory", COLOR_SECONDARY)
    kpi_card(fig.add_subplot(gs[0, 3]), "AVG TURNOVER RATIO", f"{avg_turnover:.2f}", "Units sold / stock", COLOR_PRIMARY)

    # Fast vs Slow moving products (by units sold)
    ax1 = fig.add_subplot(gs[1, :2])
    top10 = df.groupby("Product ID")["Units Sold"].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top10.values, y=top10.index, ax=ax1, palette="Blues_r")
    ax1.set_title("Top 10 Fast-Moving Products (Units Sold)", fontsize=12, color=COLOR_PRIMARY)
    ax1.set_xlabel("Units Sold")

    ax2 = fig.add_subplot(gs[1, 2:])
    bottom10 = df.groupby("Product ID")["Units Sold"].sum().sort_values().head(10)
    sns.barplot(x=bottom10.values, y=bottom10.index, ax=ax2, palette="Reds_r")
    ax2.set_title("Top 10 Slow-Moving Products (Units Sold)", fontsize=12, color=COLOR_PRIMARY)
    ax2.set_xlabel("Units Sold")

    # Stock level distribution + reorder by category
    ax3 = fig.add_subplot(gs[2, :2])
    sns.histplot(df["Stock Quantity"], bins=30, color=COLOR_SECONDARY, ax=ax3, kde=True)
    ax3.set_title("Stock Quantity Distribution", fontsize=12, color=COLOR_PRIMARY)
    ax3.set_xlabel("Stock Quantity")

    ax4 = fig.add_subplot(gs[2, 2:])
    reorder_by_cat = df[df["Reorder Flag"] == 1]["Category"].value_counts()
    sns.barplot(x=reorder_by_cat.values, y=reorder_by_cat.index, ax=ax4, palette=PALETTE)
    ax4.set_title("Reorder Alerts by Category", fontsize=12, color=COLOR_PRIMARY)
    ax4.set_xlabel("Number of Reorder Alerts")

    out_path = os.path.join(VISUALS_DIR, "inventory_dashboard.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# DASHBOARD 3: SUPPLIER DASHBOARD
# =====================================================================
def build_supplier_dashboard(df):
    print("Building supplier_dashboard.png ...")

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.55, 1.3, 1.3],
                            hspace=0.65, wspace=0.4, top=0.89, bottom=0.07, left=0.06, right=0.97)

    add_dashboard_title(fig, "Supplier Performance Dashboard",
                         "Delivery Delays | Lead Time Comparison | Reliability Scoring")

    avg_lead_time = df["Lead Time"].mean()
    worst_supplier = df.groupby("Supplier")["Supplier Delay Indicator"].mean().idxmax()
    best_supplier = df.groupby("Supplier")["Supplier Delay Indicator"].mean().idxmin()
    total_delayed_orders = df["Supplier Delay Indicator"].sum()

    kpi_card(fig.add_subplot(gs[0, 0]), "AVG LEAD TIME", f"{avg_lead_time:.1f} days", "Across all suppliers", COLOR_PRIMARY)
    kpi_card(fig.add_subplot(gs[0, 1]), "DELAYED ORDERS", f"{int(total_delayed_orders):,}", "Flagged as delayed", COLOR_DANGER)
    kpi_card(fig.add_subplot(gs[0, 2]), "BEST SUPPLIER", best_supplier, "Lowest delay rate", COLOR_SUCCESS)
    kpi_card(fig.add_subplot(gs[0, 3]), "WORST SUPPLIER", worst_supplier, "Highest delay rate", COLOR_ACCENT)

    # Supplier delay %
    ax1 = fig.add_subplot(gs[1, :2])
    delay_pct = df.groupby("Supplier")["Supplier Delay Indicator"].mean().sort_values(ascending=False) * 100
    bars = ax1.bar(delay_pct.index, delay_pct.values, color=COLOR_DANGER)
    ax1.set_title("Supplier Delay Rate (%)", fontsize=12, color=COLOR_PRIMARY)
    ax1.set_ylabel("Delay %")
    ax1.tick_params(axis="x", rotation=20)
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.1f}%",
                  ha="center", fontsize=8)

    # Lead time comparison
    ax2 = fig.add_subplot(gs[1, 2:])
    lead_comp = df.groupby("Supplier")["Lead Time"].mean().sort_values(ascending=False)
    sns.barplot(x=lead_comp.index, y=lead_comp.values, ax=ax2, palette="viridis")
    ax2.set_title("Average Lead Time by Supplier (days)", fontsize=12, color=COLOR_PRIMARY)
    ax2.tick_params(axis="x", rotation=20)
    ax2.set_ylabel("Lead Time (days)")

    # Lead time distribution by supplier (boxplot)
    ax3 = fig.add_subplot(gs[2, :2])
    sns.boxplot(data=df, x="Supplier", y="Lead Time", ax=ax3, palette=PALETTE)
    ax3.set_title("Lead Time Distribution by Supplier", fontsize=12, color=COLOR_PRIMARY)
    ax3.tick_params(axis="x", rotation=20)

    # Fulfillment time by supplier
    ax4 = fig.add_subplot(gs[2, 2:])
    fulfillment = df.groupby("Supplier")["Total Fulfillment Time"].mean().sort_values(ascending=False)
    sns.barplot(x=fulfillment.index, y=fulfillment.values, ax=ax4, palette="magma")
    ax4.set_title("Avg Total Fulfillment Time by Supplier", fontsize=12, color=COLOR_PRIMARY)
    ax4.tick_params(axis="x", rotation=20)
    ax4.set_ylabel("Days (Lead + Shipping)")

    out_path = os.path.join(VISUALS_DIR, "supplier_dashboard.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# DASHBOARD 4: SALES DASHBOARD
# =====================================================================
def build_sales_dashboard(df):
    print("Building sales_dashboard.png ...")

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.55, 1.3, 1.3],
                            hspace=0.7, wspace=0.4, top=0.90, bottom=0.06, left=0.06, right=0.97)

    add_dashboard_title(fig, "Sales & Revenue Performance Dashboard",
                         "Monthly Trends | Category Performance | Top Products")

    total_revenue = df["Revenue"].sum()
    total_units = df["Units Sold"].sum()
    avg_order_value = df["Revenue"].sum() / df.shape[0]
    top_category = df.groupby("Category")["Revenue"].sum().idxmax()

    kpi_card(fig.add_subplot(gs[0, 0]), "TOTAL REVENUE", f"${total_revenue/1e6:.2f}M", "", COLOR_PRIMARY)
    kpi_card(fig.add_subplot(gs[0, 1]), "TOTAL UNITS SOLD", f"{int(total_units):,}", "", COLOR_SECONDARY)
    kpi_card(fig.add_subplot(gs[0, 2]), "AVG ORDER VALUE", f"${avg_order_value:,.0f}", "", COLOR_ACCENT)
    kpi_card(fig.add_subplot(gs[0, 3]), "TOP CATEGORY", top_category, "By revenue", COLOR_SUCCESS)

    # Monthly revenue trend
    ax1 = fig.add_subplot(gs[1, :])
    monthly = df.groupby("Order Month")["Revenue"].sum().sort_index()
    ax1.plot(monthly.index, monthly.values, marker="o", color=COLOR_PRIMARY, linewidth=2.2, markersize=5)
    ax1.fill_between(range(len(monthly)), monthly.values, color=COLOR_PRIMARY, alpha=0.12)
    ax1.set_title("Monthly Revenue Trend", fontsize=13, color=COLOR_PRIMARY)
    ax1.tick_params(axis="x", rotation=60, labelsize=8)
    ax1.set_ylabel("Revenue ($)")

    # Category sales
    ax2 = fig.add_subplot(gs[2, :2])
    cat_sales = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    sns.barplot(x=cat_sales.values, y=cat_sales.index, ax=ax2, palette=PALETTE)
    ax2.set_title("Revenue by Category", fontsize=12, color=COLOR_PRIMARY)
    ax2.set_xlabel("Revenue ($)")

    # Top products
    ax3 = fig.add_subplot(gs[2, 2:])
    top_products = df.groupby("Product ID")["Revenue"].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax3, palette="crest")
    ax3.set_title("Top 10 Products by Revenue", fontsize=12, color=COLOR_PRIMARY)
    ax3.set_xlabel("Revenue ($)")

    out_path = os.path.join(VISUALS_DIR, "sales_dashboard.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("SUPPLY CHAIN INVENTORY OPTIMIZATION - EDA & BI DASHBOARDS")
    print("=" * 70)

    df = load_data()

    inventory_analysis(df)
    sales_analytics(df)
    supplier_analytics(df)
    warehouse_analytics(df)

    print("\n" + "=" * 70)
    print("GENERATING PROFESSIONAL DASHBOARD PNG REPORTS")
    print("=" * 70)

    build_executive_dashboard(df)
    build_inventory_dashboard(df)
    build_supplier_dashboard(df)
    build_sales_dashboard(df)

    print("\n" + "=" * 70)
    print(f"All dashboards saved successfully in: {VISUALS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
