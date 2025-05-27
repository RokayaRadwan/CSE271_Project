# Social Media and Smartphone Addiction Analysis

This project analyzes smartphone usage behavior to identify indicators of addiction and stress among users. Using machine learning and data visualization techniques, we explore how digital habits like screen time, notification frequency, and night usage influence user well-being.

## Project Objectives

- Analyze behavioral trends in mobile phone usage.
- Predict smartphone addiction and high stress risk using classification models.
- Model continuous stress levels using regression models.
- Visualize key patterns using interactive and static visualizations.
- Provide insights via an interactive Streamlit dashboard.

## Key Insights

- **High screen time**, **night usage**, and **notifications** are strongly associated with increased stress and addiction.
- **Younger users** exhibit higher risk indicators for both addiction and stress.
- **Logistic Regression** and **KNN** provided excellent classification performance (~98% and ~97% accuracy respectively).
- **Linear regression models** performed best in predicting stress level (R² ≈ 0.76), suggesting a strong linear relationship.

## 📁 Dataset

The project uses a behavioral dataset `mobile_usage_analysis.csv` with 13,589 records. Features include:
- `screen_time`, `night_usage`, `notifications`
- `age`, `gaming_time`, `installed_apps`
- Target Variables: `addiction_status` (binary), `stress_risk` (binary), `stress_level` (continuous)

## ⚙️ Methods

### Preprocessing
- No missing values.
- Target encoding (binary).
- Standard scaling for features.
- 80/20 train-test split.

### Visualization
- Boxplots, heatmaps, scatter plots, pairplots.
- Feature importance via bar charts.
- ROC curves for classifiers.
- Residual and prediction plots for regressors.

### Modeling

#### Classification
- **Logistic Regression**: High interpretability, linear boundaries.
- **K-Nearest Neighbors (KNN)**: Captures non-linear patterns.

#### Regression
- **Multiple Linear Regression**
- **Lasso Regression**
- **Polynomial Regression**
- **Random Forest Regression**

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC.
- **Regression**: R², MSE, RMSE, Residual plots.

## 📈 Results

| Model                     | Accuracy / R² | RMSE   |
|--------------------------|---------------|--------|
| Logistic Regression      | 98%           | —      |
| KNN (Stress Risk)        | 97%           | —      |
| Linear/Lasso Regression  | 0.76          | 9.93   |
| Polynomial Regression    | 0.76          | 9.93   |
| Random Forest Regression | 0.66          | 11.74  |

## 🛠️ Tools & Libraries

- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `streamlit`

## 🙌 Acknowledgements

- **Team Members**:
  - *Rokaya Radwan*: Regression models & analysis.
  - *Jana Sherif*: Classification models & evaluation.
- **Instructor**: Dr. Fatma ElShahaby

---

© 2025 Rokaya Radwan & Jana Sherif — For academic use only.
