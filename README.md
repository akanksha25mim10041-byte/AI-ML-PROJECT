
# 📊 AI-ML-PROJECT
## Academic Score Forecaster

A machine learning project that predicts student exam scores based on study habits, attendance, sleep, and other lifestyle factors. The project also includes an interactive **Streamlit web application** and **SHAP-based explainable AI** to help users understand model predictions.

---

## 📌 Overview

The Academic Score Forecaster is a supervised machine learning regression project that estimates a student's final exam score using academic and behavioral features.

The project compares four regression algorithms and provides an interactive web interface where users can enter student information and receive a predicted exam score.

The Streamlit application allows users to:
- Enter student academic and lifestyle details.
- Generate predicted exam scores.
- View model-based feature importance.
- Understand how different features influence individual predictions using SHAP.
- Interact with the machine learning model through a user-friendly interface.

---

## 📌 Problem Statement

Student academic performance is influenced by several factors, including study hours, attendance, sleep, previous scores, parental education, and access to resources.

Traditional methods of identifying struggling students may depend on delayed examination results or subjective judgment.

This project develops a machine learning-based prediction system that estimates exam scores from student-related features. The system also uses explainable AI techniques to provide insights into the factors influencing predictions.

---

## 🎯 Project Objectives

1. Predict student exam scores using machine learning.
2. Compare multiple regression algorithms.
3. Identify the best-performing regression model.
4. Develop an interactive Streamlit web application.
5. Explain individual predictions using SHAP.
6. Provide insights into the factors associated with academic performance.

---

## 📁 Project Structure

```text
academic-score-forecaster/
│
├── academic_score_forecaster.py  # Main machine learning pipeline
├── app.py                        # Streamlit web application
├── student_performance.csv       # Dataset
├── eda_plots.png                 # EDA visualizations
├── project_report.docx           # Project report
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset Summary

The dataset contains 100 student records and includes the following features:

| Feature | Description |
|---|---|
| student_id | Unique student identifier |
| gender | Gender of the student |
| age | Age of the student |
| study_hours_per_day | Average daily study hours |
| attendance_percentage | Class attendance percentage |
| sleep_hours | Average sleep per night |
| previous_score | Previous examination score |
| extracurricular | Participation in extracurricular activities |
| internet_access | Availability of internet access |
| parent_education_level | Parent's education level |
| exam_score | Final examination score (target variable) |

### Key Dataset Insights

- The dataset contains 100 student records.
- No missing values are reported in the original dataset.
- Exam scores range approximately from 50 to 95.
- Study hours vary from approximately 2 to 8 hours per day.
- Most students have attendance above 70%.

---

## 🧠 Machine Learning Concepts Used

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Label Encoding
- Feature Scaling
- Train-Test Split
- Regression Algorithms
- Model Evaluation
- Feature Importance Analysis
- Explainable AI using SHAP
- Interactive Machine Learning Deployment using Streamlit

---

## 🤖 Machine Learning Models

The project compares the following regression algorithms:

| Model | Description |
|---|---|
| Linear Regression | Baseline linear regression model |
| Decision Tree Regressor | Tree-based regression model |
| Random Forest Regressor | Ensemble of multiple decision trees |
| Gradient Boosting Regressor | Sequential boosting-based model |

### Reported Model Performance

| Model | R² Score |
|---|---:|
| Linear Regression | 0.9823 |
| Decision Tree Regressor | 0.9741 |
| Random Forest Regressor | 0.9901 |
| Gradient Boosting Regressor | 0.9887 |

The reported results identify Random Forest Regressor as the best-performing model in the original experiment.

> Note: These performance values depend on the dataset, preprocessing, train-test split, and model configuration.

---

## 🌐 Streamlit Web Application

The project includes an interactive web application developed using **Streamlit**.

The application provides a graphical interface for interacting with the machine learning model without requiring users to enter commands in the terminal.

### Streamlit Features

- User-friendly input interface.
- Input fields for student academic and lifestyle information.
- Real-time exam score prediction after submitting details.
- Predicted score displayed out of 100.
- Feature importance visualization.
- SHAP-based explanation of individual predictions.
- Interactive and accessible machine learning demonstration.

### Application Workflow

```text
User enters student details
          ↓
Input preprocessing
          ↓
Machine learning model
          ↓
Predicted exam score
          ↓
Feature importance and SHAP explanation
```

---

## 🔍 Explainable AI Using SHAP

The application integrates **SHAP (SHapley Additive exPlanations)** to improve the interpretability of machine learning predictions.

SHAP helps explain how individual input features contribute to a model's prediction.

For example, the SHAP explanation can show whether features such as:

- Study hours
- Previous examination score
- Attendance percentage
- Sleep hours
- Other student attributes

contribute positively or negatively to a particular prediction.

### Benefits of SHAP

- Improves model transparency.
- Helps users understand individual predictions.
- Identifies influential features.
- Supports explainable machine learning.
- Makes model results easier to discuss during project demonstrations.

> SHAP explanations show model associations and contributions. They do not prove that a feature directly causes a student's academic performance.

---

## 🛠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models and preprocessing |
| Streamlit | Interactive web application |
| SHAP | Explainable AI and prediction interpretation |
| Joblib | Model and preprocessing artifact management |

---

## 🔄 Project Workflow

1. **Data Loading** – Load the student performance dataset.
2. **Data Inspection** – Check data shape, data types, and missing values.
3. **Exploratory Data Analysis** – Generate visualizations to understand the dataset.
4. **Preprocessing** – Encode categorical features and scale numerical features.
5. **Train-Test Split** – Divide the dataset into training and testing sets.
6. **Model Training** – Train four regression algorithms.
7. **Model Evaluation** – Calculate MAE, RMSE, and R² scores.
8. **Best Model Selection** – Compare model performance.
9. **Streamlit Deployment** – Create an interactive prediction interface.
10. **Prediction** – Accept new student information and estimate the exam score.
11. **SHAP Explanation** – Explain how input features influence individual predictions.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/akanksha25mim10041-byte/AI-ML-PROJECT.git
cd AI-ML-PROJECT
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit shap joblib
```

Alternatively, install the dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline

```bash
python academic_score_forecaster.py
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address displayed in the terminal.

---

## 📈 Generated Visualizations

The project can generate the following visualizations:

- `eda_plots.png` — Exploratory data analysis plots.
- `model_comparison.png` — Comparison of model performance.
- `actual_vs_predicted.png` — Actual versus predicted scores.
- `feature_importance.png` — Feature importance visualization.
- SHAP plots — Explanations for individual predictions in the Streamlit application.

---

## 🎓 Sample Prediction

### Example Input

| Feature | Value |
|---|---|
| Gender | Male |
| Age | 19 |
| Study Hours | 6 hours/day |
| Attendance | 88% |
| Sleep | 7 hours |
| Previous Score | 75 |
| Extracurricular | Yes |
| Internet Access | Yes |
| Parent Education | Graduate |

The original project documentation reports an illustrative predicted score of **81.4/100**.

> The exact prediction displayed by the Streamlit application may vary depending on the trained model and preprocessing configuration.

---

## 🔮 Future Enhancements

- Hyperparameter tuning using GridSearchCV.
- K-fold cross-validation.
- Training on a larger dataset.
- Model artifact loading instead of retraining.
- Cloud deployment of the Streamlit application.
- Improved SHAP visualizations.
- Pass/fail and grade classification.
- Student performance trend analysis.
- Automated academic support recommendations.

---

## ✨ Key Highlights

- Complete machine learning regression pipeline.
- Comparison of four regression models.
- Interactive Streamlit prediction application.
- SHAP-based explainable AI integration.
- Feature importance analysis.
- Automated data visualization.
- Easy-to-use interface for demonstrating machine learning predictions.

---

## 👤 Author

**Akanksha Chauhan**
