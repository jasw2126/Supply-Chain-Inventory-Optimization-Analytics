"""
=====================================================================
SUPPLY CHAIN INVENTORY OPTIMIZATION ANALYTICS
Module 3: Machine Learning - Demand Forecasting & Inventory Risk
=====================================================================
Author : Data Analytics Portfolio Project
Purpose: Train and evaluate two ML models on the cleaned supply
         chain dataset:
           1. Random Forest Regressor  -> Demand Forecasting
           2. Random Forest Classifier -> Inventory Risk Classification
         Generates evaluation metrics and professional PNG reports.
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_supply_chain_data.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Styling (consistent with eda_analysis.py dashboards)
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"

COLOR_PRIMARY = "#1F4E79"
COLOR_SECONDARY = "#2E86AB"
COLOR_ACCENT = "#F39C12"
COLOR_DANGER = "#E74C3C"
COLOR_SUCCESS = "#27AE60"
COLOR_GREY = "#7F8C8D"

RANDOM_STATE = 42


def load_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def kpi_card(ax, title, value, subtitle="", color=COLOR_PRIMARY):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                          facecolor=color, edgecolor="none", zorder=1, alpha=0.95)
    ax.add_patch(card)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=18,
            fontweight="bold", color="white", transform=ax.transAxes, zorder=2)
    ax.text(0.5, 0.30, title, ha="center", va="center", fontsize=10,
            color="white", transform=ax.transAxes, zorder=2)
    if subtitle:
        ax.text(0.5, 0.12, subtitle, ha="center", va="center", fontsize=8,
                color="#EAEAEA", transform=ax.transAxes, zorder=2)


# =====================================================================
# MODEL 1: DEMAND FORECASTING (RANDOM FOREST REGRESSOR)
# =====================================================================
def build_demand_forecasting_model(df):
    print("=" * 70)
    print("MODEL 1: DEMAND FORECASTING (Random Forest Regressor)")
    print("=" * 70)

    data = df.copy()

    # Target: Units Sold -> used as a proxy for realized product demand
    target_col = "Units Sold"

    feature_cols = [
        "Stock Quantity", "Lead Time", "Shipping Time", "Inventory Cost",
        "Unit Cost", "Unit Price", "Total Fulfillment Time",
        "Category", "Warehouse", "Supplier", "Order Year"
    ]

    data = data.dropna(subset=feature_cols + [target_col])

    # Encode categorical variables
    encoders = {}
    data_enc = data.copy()
    for col in ["Category", "Warehouse", "Supplier"]:
        le = LabelEncoder()
        data_enc[col] = le.fit_transform(data_enc[col])
        encoders[col] = le

    X = data_enc[feature_cols]
    y = data_enc[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Training samples: {X_train.shape[0]}  |  Testing samples: {X_test.shape[0]}")

    model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_split=4,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Regression Evaluation Metrics ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # Save model + encoders
    joblib.dump(model, os.path.join(MODELS_DIR, "demand_forecast_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "demand_forecast_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "demand_forecast_features.pkl"))
    print(f"\nModel saved to: {os.path.join(MODELS_DIR, 'demand_forecast_model.pkl')}")

    results = {
        "model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "feature_cols": feature_cols, "mae": mae, "rmse": rmse, "r2": r2
    }
    return results


# =====================================================================
# MODEL 2: INVENTORY RISK CLASSIFICATION (RANDOM FOREST CLASSIFIER)
# =====================================================================
def build_inventory_risk_model(df):
    print("\n" + "=" * 70)
    print("MODEL 2: INVENTORY RISK CLASSIFICATION (Random Forest Classifier)")
    print("=" * 70)

    data = df.copy()

    # Target: 1 = Low Stock Risk (Understocked), 0 = Safe Stock
    data["Risk Label"] = np.where(data["Stock Risk"] == "Understocked", 1, 0)

    feature_cols = [
        "Stock Quantity", "Demand Forecast", "Lead Time", "Shipping Time",
        "Units Sold", "Inventory Turnover Ratio", "Total Fulfillment Time",
        "Category", "Warehouse", "Supplier", "Supplier Delay Indicator"
    ]

    data = data.dropna(subset=feature_cols + ["Risk Label"])

    encoders = {}
    data_enc = data.copy()
    for col in ["Category", "Warehouse", "Supplier"]:
        le = LabelEncoder()
        data_enc[col] = le.fit_transform(data_enc[col])
        encoders[col] = le

    X = data_enc[feature_cols]
    y = data_enc["Risk Label"]

    print(f"\nClass distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining samples: {X_train.shape[0]}  |  Testing samples: {X_test.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Classification Evaluation Metrics ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Safe Stock", "Low Stock Risk"]))

    joblib.dump(model, os.path.join(MODELS_DIR, "inventory_risk_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "inventory_risk_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "inventory_risk_features.pkl"))
    print(f"\nModel saved to: {os.path.join(MODELS_DIR, 'inventory_risk_model.pkl')}")

    results = {
        "model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "feature_cols": feature_cols, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1
    }
    return results


# =====================================================================
# VISUAL 1: FEATURE IMPORTANCE DASHBOARD (both models)
# =====================================================================
def build_feature_importance_dashboard(reg_results, clf_results):
    print("\nBuilding ml_feature_importance.png ...")

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4, top=0.85, bottom=0.15, left=0.08, right=0.97)

    fig.suptitle("Machine Learning - Feature Importance Report", fontsize=18,
                 fontweight="bold", color=COLOR_PRIMARY, x=0.02, ha="left")
    fig.text(0.02, 0.90, "Random Forest Regressor (Demand) & Classifier (Inventory Risk)",
              fontsize=11, color=COLOR_GREY, ha="left")

    # Regression feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    reg_importances = pd.Series(
        reg_results["model"].feature_importances_, index=reg_results["feature_cols"]
    ).sort_values(ascending=True)
    ax1.barh(reg_importances.index, reg_importances.values, color=COLOR_SECONDARY)
    ax1.set_title("Demand Forecasting Model\n(Feature Importance)", fontsize=12, color=COLOR_PRIMARY)
    ax1.set_xlabel("Importance Score")

    # Classification feature importance
    ax2 = fig.add_subplot(gs[0, 1])
    clf_importances = pd.Series(
        clf_results["model"].feature_importances_, index=clf_results["feature_cols"]
    ).sort_values(ascending=True)
    ax2.barh(clf_importances.index, clf_importances.values, color=COLOR_ACCENT)
    ax2.set_title("Inventory Risk Classification Model\n(Feature Importance)", fontsize=12, color=COLOR_PRIMARY)
    ax2.set_xlabel("Importance Score")

    out_path = os.path.join(VISUALS_DIR, "ml_feature_importance.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# VISUAL 2: PREDICTION COMPARISON DASHBOARD (both models)
# =====================================================================
def build_prediction_comparison_dashboard(reg_results, clf_results):
    print("Building ml_prediction_comparison.png ...")

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[0.45, 1.4],
                            hspace=0.55, wspace=0.4, top=0.88, bottom=0.08, left=0.06, right=0.97)

    fig.suptitle("Machine Learning - Model Performance & Prediction Comparison", fontsize=18,
                 fontweight="bold", color=COLOR_PRIMARY, x=0.02, ha="left")
    fig.text(0.02, 0.925, "Demand Forecasting (Regression) & Inventory Risk Classification",
              fontsize=11, color=COLOR_GREY, ha="left")

    # KPI row
    kpi_card(fig.add_subplot(gs[0, 0]), "R2 SCORE", f"{reg_results['r2']:.3f}", "Demand model", COLOR_PRIMARY)
    kpi_card(fig.add_subplot(gs[0, 1]), "RMSE", f"{reg_results['rmse']:.1f}", "Demand model", COLOR_SECONDARY)
    kpi_card(fig.add_subplot(gs[0, 2]), "CLASSIFIER ACCURACY", f"{clf_results['accuracy']*100:.1f}%", "Risk model", COLOR_SUCCESS)

    # Actual vs Predicted scatter (Regression)
    ax1 = fig.add_subplot(gs[1, 0])
    sample_idx = np.random.choice(len(reg_results["y_test"]), size=min(500, len(reg_results["y_test"])), replace=False)
    y_test_arr = np.array(reg_results["y_test"])[sample_idx]
    y_pred_arr = np.array(reg_results["y_pred"])[sample_idx]
    ax1.scatter(y_test_arr, y_pred_arr, alpha=0.4, color=COLOR_SECONDARY, s=18)
    lims = [min(y_test_arr.min(), y_pred_arr.min()), max(y_test_arr.max(), y_pred_arr.max())]
    ax1.plot(lims, lims, color=COLOR_DANGER, linestyle="--", linewidth=1.5, label="Ideal Prediction")
    ax1.set_title("Actual vs Predicted Demand", fontsize=12, color=COLOR_PRIMARY)
    ax1.set_xlabel("Actual Units Sold")
    ax1.set_ylabel("Predicted Units Sold")
    ax1.legend(fontsize=8)

    # Residual distribution (Regression)
    ax2 = fig.add_subplot(gs[1, 1])
    residuals = np.array(reg_results["y_test"]) - np.array(reg_results["y_pred"])
    sns.histplot(residuals, bins=30, color=COLOR_ACCENT, kde=True, ax=ax2)
    ax2.axvline(0, color=COLOR_DANGER, linestyle="--", linewidth=1.5)
    ax2.set_title("Prediction Error Distribution (Residuals)", fontsize=12, color=COLOR_PRIMARY)
    ax2.set_xlabel("Residual (Actual - Predicted)")

    # Confusion Matrix (Classification)
    ax3 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(clf_results["y_test"], clf_results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3, cbar=False,
                xticklabels=["Safe Stock", "Low Stock Risk"],
                yticklabels=["Safe Stock", "Low Stock Risk"])
    ax3.set_title("Inventory Risk - Confusion Matrix", fontsize=12, color=COLOR_PRIMARY)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    out_path = os.path.join(VISUALS_DIR, "ml_prediction_comparison.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("SUPPLY CHAIN INVENTORY OPTIMIZATION - MACHINE LEARNING MODULE")
    print("=" * 70)

    df = load_data()

    reg_results = build_demand_forecasting_model(df)
    clf_results = build_inventory_risk_model(df)

    print("\n" + "=" * 70)
    print("GENERATING ML PERFORMANCE PNG REPORTS")
    print("=" * 70)

    build_feature_importance_dashboard(reg_results, clf_results)
    build_prediction_comparison_dashboard(reg_results, clf_results)

    print("\n" + "=" * 70)
    print("MACHINE LEARNING SUMMARY")
    print("=" * 70)
    print(f"Demand Forecasting  -> MAE: {reg_results['mae']:.2f} | RMSE: {reg_results['rmse']:.2f} | R2: {reg_results['r2']:.4f}")
    print(f"Inventory Risk      -> Accuracy: {clf_results['accuracy']:.4f} | Precision: {clf_results['precision']:.4f} | "
          f"Recall: {clf_results['recall']:.4f} | F1: {clf_results['f1']:.4f}")
    print("=" * 70)
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Visual reports saved in: {VISUALS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
