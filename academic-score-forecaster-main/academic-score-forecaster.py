# ============================================================
#  Academic Score Forecaster
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────

def load_data(filepath="student_performance.csv"):
    df = pd.read_csv(filepath)
    print("=" * 55)
    print("       ACADEMIC SCORE FORECASTER")
    print("=" * 55)
    print(f"\n📂 Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📊 Column Info:")
    print(df.dtypes)
    print(f"\n📈 Basic Statistics:")
    print(df.describe())
    print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def perform_eda(df):
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Academic Performance – EDA Dashboard", fontsize=16, fontweight="bold")

    # Distribution of exam scores
    axes[0, 0].hist(df["exam_score"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Distribution of Exam Scores")
    axes[0, 0].set_xlabel("Exam Score")
    axes[0, 0].set_ylabel("Frequency")

    # Study hours vs Exam Score
    axes[0, 1].scatter(df["study_hours_per_day"], df["exam_score"],
                       color="#DD8452", alpha=0.7)
    axes[0, 1].set_title("Study Hours vs Exam Score")
    axes[0, 1].set_xlabel("Study Hours / Day")
    axes[0, 1].set_ylabel("Exam Score")

    # Attendance vs Exam Score
    axes[0, 2].scatter(df["attendance_percentage"], df["exam_score"],
                       color="#55A868", alpha=0.7)
    axes[0, 2].set_title("Attendance % vs Exam Score")
    axes[0, 2].set_xlabel("Attendance (%)")
    axes[0, 2].set_ylabel("Exam Score")

    # Avg score by parent education
    avg_by_edu = df.groupby("parent_education_level")["exam_score"].mean().sort_values()
    axes[1, 0].barh(avg_by_edu.index, avg_by_edu.values, color="#C44E52")
    axes[1, 0].set_title("Avg Score by Parent Education")
    axes[1, 0].set_xlabel("Average Exam Score")

    # Avg score by gender
    avg_by_gender = df.groupby("gender")["exam_score"].mean()
    axes[1, 1].bar(avg_by_gender.index, avg_by_gender.values,
                   color=["#4C72B0", "#DD8452"])
    axes[1, 1].set_title("Avg Score by Gender")
    axes[1, 1].set_ylabel("Average Exam Score")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number]).drop("student_id", axis=1)
    corr = numeric_df.corr()
    sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5)
    axes[1, 2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    print("✅ EDA plots saved → eda_plots.png")
    plt.show()

    # Print correlation with target
    print("\n📌 Correlation with Exam Score:")
    print(corr["exam_score"].sort_values(ascending=False))


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n" + "=" * 55)
    print("  PREPROCESSING")
    print("=" * 55)

    df = df.drop("student_id", axis=1)

    # Encode categorical columns
    le = LabelEncoder()
    categorical_cols = ["gender", "extracurricular", "internet_access", "parent_education_level"]
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"  ✔ Encoded: {col}")

    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train size : {X_train.shape[0]} samples")
    print(f"  Test size  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, X_scaled, y


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 55)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 55)

    models = {
        "Linear Regression":       LinearRegression(),
        "Decision Tree":           DecisionTreeRegressor(random_state=42),
        "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2, "predictions": y_pred}
        print(f"\n  🤖 {name}")
        print(f"     MAE  : {mae:.2f}")
        print(f"     RMSE : {rmse:.2f}")
        print(f"     R²   : {r2:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. VISUALIZE MODEL COMPARISON & BEST MODEL
# ─────────────────────────────────────────────

def visualize_results(results, X_test, y_test, X_scaled, y):
    print("\n" + "=" * 55)
    print("  RESULTS & VISUALIZATIONS")
    print("=" * 55)

    names  = list(results.keys())
    r2s    = [results[n]["R2"]   for n in names]
    maes   = [results[n]["MAE"]  for n in names]
    rmses  = [results[n]["RMSE"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    axes[0].bar(names, r2s, color=colors)
    axes[0].set_title("R² Score (higher = better)")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, maes, color=colors)
    axes[1].set_title("MAE (lower = better)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(names, rmses, color=colors)
    axes[2].set_title("RMSE (lower = better)")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("✅ Model comparison plot saved → model_comparison.png")
    plt.show()

    # Best model: highest R²
    best_name = max(results, key=lambda n: results[n]["R2"])
    best      = results[best_name]
    print(f"\n🏆 Best Model: {best_name}  (R² = {best['R2']:.4f})")

    # Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best["predictions"], alpha=0.7, color="#4C72B0")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"Actual vs Predicted – {best_name}")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    print("✅ Actual vs Predicted plot saved → actual_vs_predicted.png")
    plt.show()

    # Feature importance (if tree-based)
    if hasattr(best["model"], "feature_importances_"):
        feat_imp = pd.Series(
            best["model"].feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=True)
        plt.figure(figsize=(8, 5))
        feat_imp.plot(kind="barh", color="#55A868")
        plt.title(f"Feature Importance – {best_name}")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
        print("✅ Feature importance plot saved → feature_importance.png")
        plt.show()

    return best_name, best["model"]


# ─────────────────────────────────────────────
# 6. PREDICT FOR A NEW STUDENT
# ─────────────────────────────────────────────

def predict_new_student(model):
    print("\n" + "=" * 55)
    print("  PREDICT FOR A NEW STUDENT")
    print("=" * 55)

    # Example student (features must match training order)
    new_student = pd.DataFrame([{
        "gender":                  1,    # 1=Male, 0=Female
        "age":                     19,
        "study_hours_per_day":     6,
        "attendance_percentage":   88,
        "sleep_hours":             7,
        "previous_score":          75,
        "extracurricular":         1,    # 1=Yes, 0=No
        "internet_access":         1,
        "parent_education_level":  1,    # encoded
    }])

    prediction = model.predict(new_student)[0]
    print(f"\n  🎓 Predicted Exam Score: {prediction:.1f} / 100")
    return prediction


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df                                        = load_data("student-performance.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test, X_s, y = preprocess(df)
    results                                   = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name, best_model                     = visualize_results(results, X_test, y_test, X_s, y)
    predict_new_student(best_model)

    print("\n" + "=" * 55)
    print("  ✅ ALL DONE! Check the generated PNG plots.")
    print("=" * 55)
