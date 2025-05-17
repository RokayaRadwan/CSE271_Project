# FINAL DASHBOARD.PY WITH FINAL TOUCHES
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classification_models import (
    load_classification_data, scale_features, train_logistic_model,
    train_knn_model, evaluate_model, plot_confusion_matrix,
    plot_roc_curve, plot_classification_heatmap, plot_pairplot
)
from regression_models import (
    load_regression_data, train_regression_model,
    plot_regression_heatmap, plot_regression_scatter
)
from sklearn.metrics import r2_score, mean_squared_error

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# NAVIGATION
st.sidebar.title("Navigation")
if st.session_state.page == 'Home':
    st.title("👩‍🏫 Welcome Dr. Fatma!")
    st.subheader("📘 Data Science Methodologies Project")
    option = st.radio("Choose Technique", ["Classification", "Regression"])
    if st.button("✅ Proceed"):
        st.session_state.page = option
else:
    if st.sidebar.button("🔙 Back to Home"):
        st.session_state.page = 'Home'

# CLASSIFICATION PAGE
if st.session_state.page == 'Classification':
    st.title("🧠 Classification")
    model_type = st.sidebar.radio("Choose Model", ["Logistic Regression", "KNN"])
    df = load_classification_data()
    features = ['daily_screen_time', 'notifications_number', 'app_sessions', 'night_usage', 'age']
    target = 'addiction_status' if model_type == 'Logistic Regression' else 'stress_risk'

    X_scaled, scaler = scale_features(df, features)
    if model_type == 'Logistic Regression':
        model, X_train, X_test, y_train, y_test = train_logistic_model(X_scaled, df[target])
    else:
        model, X_train, X_test, y_train, y_test = train_knn_model(X_scaled, df[target])

    y_pred, y_score, acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)

    st.subheader("Metrics")
    st.write(f"Accuracy: {acc:.2f} Precision: {prec:.2f} Recall: {rec:.2f} F1: {f1:.2f}")

    if st.checkbox("Show Confusion Matrix"):
        labels = ['Not Addicted', 'Addicted'] if model_type == 'Logistic Regression' else ['Low Stress Risk', 'High Stress Risk']
        fig = plot_confusion_matrix(y_test, y_pred, labels)
        st.pyplot(fig)

    if st.checkbox("Show ROC Curve"):
        fig = plot_roc_curve(y_test, y_score, model_type)
        st.pyplot(fig)

    st.subheader("🔢 Predict")
    inputs = [
        st.slider("Daily Screen Time", 1, 12, 6),
        st.slider("Notifications Number", 10, 150, 50),
        st.slider("App Sessions", 1, 100, 20),
        st.slider("Night Usage", 1, 10, 5),
        st.slider("Age", 10, 80, 30)
    ]
    if st.button("🔮 Predict"):
        scaled = scaler.transform([inputs])
        result = model.predict(scaled)[0]
        if model_type == 'Logistic Regression':
            label = "Addicted" if result == 1 else "Not Addicted"
        else:
            label = "High Stress Risk" if result == 1 else "Low Stress Risk"
        st.success(f"Prediction: {label}")

    st.subheader("📊 Visualizations")
    viz = st.selectbox("Choose", ["Heatmap", "Boxplot", "Pairplot"])
    if viz == "Heatmap":
        fig = plot_classification_heatmap(df, features)
        st.pyplot(fig)
    elif viz == "Boxplot":
        f = st.selectbox("Feature", features)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[target], y=df[f], ax=ax)
        st.pyplot(fig)
    elif viz == "Pairplot":
        fig = plot_pairplot(df, target, features)
        st.pyplot(fig)

# REGRESSION PAGE
elif st.session_state.page == 'Regression':
    st.title("📈 Regression")
    model_type = st.sidebar.radio("Model", ["Linear", "Lasso", "Polynomial", "Random Forest"])
    df, X, y = load_regression_data()
    model, X_test, y_test, y_pred = train_regression_model(model_type, X, y)

    st.subheader("Metrics")
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    st.subheader("🔢 Predict")
    inputs = [
        st.number_input("Daily Screen Time", float(df['daily_screen_time'].min()), float(df['daily_screen_time'].max()), float(df['daily_screen_time'].mean())),
        st.number_input("Night Usage", float(df['night_usage'].min()), float(df['night_usage'].max()), float(df['night_usage'].mean()))
    ]
    if st.button("🔮 Predict Regression"):
        input_df = pd.DataFrame([inputs], columns=['daily_screen_time', 'night_usage'])
        if model_type == 'Polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2)
            input_poly = poly.fit_transform(input_df)
            pred = model.predict(input_poly)[0]
        else:
            pred = model.predict(input_df)[0]
        st.success(f"Predicted Stress Level: {pred:.2f}")

    st.subheader("📊 Heatmap")
    fig = plot_regression_heatmap(df)
    st.pyplot(fig)

    st.subheader("📊 Night vs Screen Time (colored by stress)")
    fig = plot_regression_scatter(df)
    st.pyplot(fig)

    st.subheader("📊 Daily Screen Time vs Stress Level")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['daily_screen_time'], y=df['stress_level'], ax=ax)
    ax.set_xlabel("Daily Screen Time")
    ax.set_ylabel("Stress Level")
    ax.set_title("Relationship between Screen Time and Stress")
    st.pyplot(fig)
