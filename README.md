# 🩺 SugerMetrics — AI-Powered Diabetes Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-10%20Models-green?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> 🔬 An AI-powered early-diabetes prediction system trained on **10 machine learning models**, wrapped in an intuitive **Streamlit** web application for real-time health assessment.

## DataSet Used:
Dataset: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 🧠 About the Project

**SugerMetrics** is an end-to-end machine learning project that predicts the likelihood of diabetes in a patient based on key health metrics such as glucose level, BMI, blood pressure, insulin, and more.

The project benchmarks **10 different ML models** to identify the best-performing algorithm, then deploys it through a clean, interactive **Streamlit** web app where users can enter their health data and get an instant prediction.

---

## ✨ Features

- ✅ Trained & evaluated **10 ML models** for best accuracy
- ✅ Interactive **Streamlit** web interface — no coding required
- ✅ Real-time diabetes risk prediction
- ✅ Model comparison & performance metrics dashboard
- ✅ Clean data preprocessing pipeline
- ✅ Feature importance visualization
- ✅ Lightweight & easy to deploy locally

---

## 🤖 ML Models Used

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Tree-based |
| 3 | Random Forest | Ensemble |
| 4 | Gradient Boosting | Ensemble |
| 5 | XGBoost | Boosting |
| 6 | K-Nearest Neighbors (KNN) | Instance-based |
| 7 | Support Vector Machine (SVM) | Kernel-based |
| 8 | Naive Bayes | Probabilistic |
| 9 | AdaBoost | Boosting |
| 10 | Extra Trees Classifier | Ensemble |

## 🎬 Demo

> 🚀 **Live App:** _[Add your Streamlit Cloud / Hugging Face link here]_

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web app framework |
| **Scikit-learn** | ML model training & evaluation |
| **XGBoost** | Gradient boosting model |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib / Seaborn** | Data visualization |
| **Joblib / Pickle** | Model serialization |

## 👨‍💻 Author

**Husen Navsariwala**
- GitHub: [Husen](https://github.com/Hussain1u2)
- LinkedIn: [Husen](https://linkedin.com/in/husennavsariwala)

---
| Step | Action                                                                                                         | Output / Notes                                                                         |
| ---- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1    | Download the `.ipynb` notebook and `app.py` script, including all required libraries.                          | Ensure your environment has every dependency installed.                                |
| 2    | Run the `.ipynb` file in your IDE (Anaconda Navigator is the safest choice; VS Code or PyCharm also work).     | The notebook executes preprocessing, training, evaluation, and export routines.        |
| 3    | After execution, the notebook generates **3 `.pkl` files**, **multiple graph images**, and **2 `.csv` files**. | These artifacts are used by the Streamlit/FastAPI/Flask app (depending on your setup). |
| 4    | Run `app.py` after all files are generated.                                                                    | The app loads the `.pkl` models and datasets to produce predictions and UI visuals.    |


<p align="center">Made with ❤️ and Python | ⭐ Star this repo if you found it helpful!</p>
