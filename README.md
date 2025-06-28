# 🧠 Employee Health Risk Prediction System

This project is a machine learning-based web application developed to predict employee health risks using clinical and lifestyle data. It combines robust data preprocessing, predictive modeling, and an interactive web interface to provide personalized health assessments. Designed with scalability, interpretability, and user experience in mind, this system serves as a practical tool for health monitoring in workplace environments.

## 🚀 Key Features

- **Predictive Modeling**: Supervised machine learning models trained on anonymized health records to identify potential health risks.
- **Interactive Web Interface**: Developed using **Streamlit**, enabling real-time input, prediction, and health feedback for users.
- **Explainability**: SHAP values used to interpret and visualize model outputs, providing transparency and trust.
- **Cloud Deployment**: Hosted on **AWS EC2** for secure and scalable access, demonstrating full-stack MLOps integration.
- **Data Validation**: Implements real-time validation for input features, with strict checks against clinical reference ranges.

---

## 🛠️ Technologies & Tools

| Category | Technologies |
|----------|--------------|
| **Languages** | Python, HTML |
| **Libraries & Frameworks** | Pandas, NumPy, Scikit-learn, SHAP, Streamlit |
| **Machine Learning** | Random Forest, XGBoost, Logistic Regression (Grid Search CV used) |
| **Web Development** | Streamlit for frontend + backend integration |
| **Deployment** | AWS EC2, MySQL, SQLAlchemy |
| **Database** | MySQL for user and prediction storage |
| **Version Control** | Git, GitHub |

---

## 📊 Machine Learning Pipeline

1. **Data Cleaning**: Removal of outliers, normalization, and handling of missing values.
2. **Feature Engineering**: Health metrics (e.g., BMI, blood pressure) derived from user inputs.
3. **Model Training**: Evaluation of multiple models using accuracy, ROC AUC, and interpretability.
4. **Interpretability**: SHAP values explain each prediction, allowing meaningful medical insights.
5. **Prediction Output**: Real-time risk level feedback, categorized and color-coded (Low/Moderate/High Risk).

---

## 📈 Use Case & Impact

- Designed for **occupational health monitoring**: HR departments or wellness teams can assess and track employee health risks.
- Applicable in **academic research** related to **preventive medicine**, **AI in healthcare**, or **risk assessment models**.
- Demonstrates practical use of **MLOps**, **end-to-end data pipelines**, and **clinical validation**.

---

## 🧑‍💻 Author

**Samindi Edirisooriya**  
Final-year undergraduate – B.Sc. in Computer Engineering  
[LinkedIn](https://www.linkedin.com/in/samindi-edirisooriya) | [GitHub](https://github.com/samindi-ed)

---
