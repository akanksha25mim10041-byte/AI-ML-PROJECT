# 📊 Academic Score Forecaster

A machine learning project that predicts student exam scores based on study habits, attendance, sleep, and other lifestyle factors.

---

## 📌 Overview

This project builds a **supervised machine learning regression model** to estimate a student's final exam score using academic and behavioral features. It helps educators identify at-risk students early, personalize interventions, and understand which factors most influence academic success.

The system is implemented in Python using popular data science libraries and compares four regression algorithms to find the best performer. The workflow includes data loading, exploratory data analysis (EDA), preprocessing (encoding, scaling), train‑test splitting, model training, evaluation, and visualization of results.

---

## 📌 Problem Statement

Student academic performance is influenced by many factors beyond raw intelligence — study hours, attendance, sleep, parental education, and access to resources all play a role. Traditional methods of identifying struggling students often rely on subjective judgment or delayed test scores, making timely intervention difficult.

This project solves that by creating a **predictive model** that estimates a student's exam score from these features, helping educators identify at-risk students early and provide data‑driven academic support.

---

## 📁 Project Structure

academic-score-forecaster/


├── academic_score_forecaster.py # Main ML pipeline

├── student_performance.csv # Dataset (100 student records)

├── eda_plots.png # Generated EDA visualizations

├──project_report.docx # project report document

├── requirements.txt # Dependencies and requirements

└── README.md # This file


---

## 📊 Dataset Summary

The dataset (`student_performance.csv`) contains **100 student records** with the following features:

| Column | Description |
|---|---|
| `student_id` | Unique student identifier |
| `gender` | Male / Female |
| `age` | Age of the student (18–22) |
| `study_hours_per_day` | Average daily study hours |
| `attendance_percentage` | Class attendance (%) |
| `sleep_hours` | Average sleep per night |
| `previous_score` | Score in previous exam |
| `extracurricular` | Participates in activities (Yes/No) |
| `internet_access` | Has internet access (Yes/No) |
| `parent_education_level` | High School / Graduate / Postgraduate |
| `exam_score` | **Target variable** – Final exam score |

**Key insights from the dataset:**
- No missing values across any column.
- Exam scores range approximately from 50 to 95.
- Study hours vary from 2 to 8 hours per day.
- Most students have attendance above 70%.

---

## 🧠 ML Concepts Used

- Exploratory Data Analysis (EDA)
- Label Encoding & Feature Scaling
- Train/Test Split
- Regression Models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor ⭐ (best performer)
  - Gradient Boosting Regressor
- Model Evaluation: MAE, RMSE, R² Score
- Feature Importance Analysis

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML models & evaluation |

---

## 🔄 Workflow (Step‑by‑Step Explanation)

The project follows a standard machine learning pipeline:

1. **Data Loading** – Load `student_performance.csv` into a pandas DataFrame.
2. **Data Inspection** – Check shape, data types, missing values, and summary statistics.
3. **Exploratory Data Analysis (EDA)** – Generate histograms, scatter plots, correlation heatmap, and bar charts to understand feature relationships.
4. **Preprocessing** – Drop `student_id`, apply label encoding to categorical columns, scale numerical features using `StandardScaler`.
5. **Train/Test Split** – Split data into 80% training and 20% testing sets (fixed random seed for reproducibility).
6. **Model Training** – Train four regression models on the training set.
7. **Model Evaluation** – Compute MAE, RMSE, and R² on the test set for each model.
8. **Best Model Selection** – Identify the model with the highest R² score.
9. **Visualization** – Generate model comparison charts, actual vs. predicted scatter plot, and feature importance bar chart.
10. **Prediction** – Accept new student data and output a predicted exam score using the best model.

---

## 🤖 Models Implemented

| Model | Description | Performance (R²) |
|-------|-------------|------------------|
| Linear Regression | Baseline linear model | 0.9823 |
| Decision Tree Regressor | Tree‑based model splitting on features | 0.9741 |
| Random Forest Regressor | Ensemble of decision trees | **0.9901** (Best) |
| Gradient Boosting Regressor | Sequential tree‑based boosting | 0.9887 |

**Why Random Forest performed best:**  
- Handles non‑linear relationships automatically.  
- Reduces overfitting compared to a single decision tree.  
- Provides feature importance scores for interpretability.

---

## 📈 Results

- **Best Model:** Random Forest Regressor  
- **R² Score:** 0.9901  
- **MAE:** Low (exact value shown on run)  
- **RMSE:** Low

**Key Insight:**  
Study hours and previous exam score are the strongest predictors of final exam performance. Attendance and parental education also show significant correlation.

---

## 📊 Sample Prediction

| Input Features | Predicted Exam Score |
|----------------|----------------------|
| Male, 19 years, 6 study hours/day, 88% attendance, 7h sleep, previous score 75, extracurricular Yes, internet Yes, parent education Graduate | **81.4 / 100** |

More sample scenarios (illustrative):
- Low study hours (2h/day), 65% attendance → predicted score ~54
- High study hours (8h/day), 95% attendance → predicted score ~92

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PariRaghuwanshi1906/academic-score-forecaster/
cd academic-score-forecaster

2. Install Dependencies (Required Packages)
The project requires the following Python libraries:

pandas – data loading and manipulation

numpy – numerical operations

matplotlib – plotting

seaborn – enhanced visualizations

scikit-learn – machine learning models and metrics

Install all at once:
pip install pandas numpy matplotlib seaborn scikit-learn

Python 3.8+ recommended. Using a virtual environment is advised.

3. Run the Project
python academic_score_forecaster.py

---


## 📈 Sample Output (Console)

======================================================
       ACADEMIC SCORE FORECASTER
======================================================

📂 Dataset Shape : 100 rows × 11 columns

  🤖 Linear Regression     → R²: 0.9823
  🤖 Decision Tree         → R²: 0.9741
  🤖 Random Forest         → R²: 0.9901  ← Best
  🤖 Gradient Boosting     → R²: 0.9887

🏆 Best Model: Random Forest  (R² = 0.9901)

  🎓 Predicted Exam Score: 81.4 / 100

Four plots are auto-saved:

eda_plots.png — distributions, scatter plots, heatmap

model_comparison.png — R², MAE, RMSE side-by-side

actual_vs_predicted.png — best model accuracy

feature_importance.png — top contributing features

---


## 🔍 Key Findings
Study hours and previous score are the strongest predictors of exam performance.

Attendance percentage is highly correlated with final score.

Students with postgraduate parents score consistently higher on average.

Random Forest outperforms other models with R² > 0.99 on this dataset.
---


## 🚀 How to Predict for a New Student
Edit the predict_new_student() function in the script:

new_student = pd.DataFrame([{
    "gender":                 1,    # 1=Male, 0=Female
    "age":                    19,
    "study_hours_per_day":    6,
    "attendance_percentage":  88,
    "sleep_hours":            7,
    "previous_score":         75,
    "extracurricular":        1,    # 1=Yes, 0=No
    "internet_access":        1,
    "parent_education_level": 1,
}])

Run the script again or call the predict_new_student() function.

---

##🔮 Future Enhancements
Hyperparameter tuning using GridSearchCV for higher accuracy.

Cross‑validation (k‑fold) to reduce overfitting.

Larger dataset with more diverse student populations.

Web deployment using Streamlit or Flask for easy educator access.

Explainable AI (SHAP/LIME) to interpret individual predictions.

Classification alternative (pass/fail or grade brackets).

---


## ✨ Key Highlights
Clean and structured ML pipeline.

Comparison of four regression models.

Strong predictive performance (R² > 0.99).

Feature importance analysis for interpretability.

Automatic generation of publication‑ready plots.

Easy to extend and modify.

👤 Author
**Suhani Singh Parihar**
