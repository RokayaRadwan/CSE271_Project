
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_regression_data():
    df = pd.read_csv('mobile_usage_analysis.csv')
    features = ['daily_screen_time', 'night_usage']
    X = df[features]
    y = df['stress_level']
    return df, X, y

def train_regression_model(model_type, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'Linear':
        model = LinearRegression()
    elif model_type == 'Lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'Polynomial':
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def plot_regression_heatmap(df):
    corr_series = df.select_dtypes(include=np.number).corr()['stress_level'].drop('stress_level').sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(4, len(corr_series) * 0.5))
    sns.heatmap(corr_series.to_frame(), annot=True, cmap='coolwarm', ax=ax, cbar=False)
    return fig

def plot_regression_scatter(df):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['daily_screen_time'], df['night_usage'], c=df['stress_level'], cmap='viridis')
    ax.set_xlabel("Daily Screen Time")
    ax.set_ylabel("Night Usage")
    legend = ax.legend(*scatter.legend_elements(), title="Stress Level")
    ax.add_artist(legend)
    return fig
