
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

def load_classification_data():
    df = pd.read_csv('mobile_addiction.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df[['daily_screen_time', 'stress_level', 'night_usage', 'productivity_hours']] = df[['daily_screen_time', 'stress_level', 'night_usage', 'productivity_hours']].round(2)
    df.rename(columns={'addicted': 'addiction_status', 'stress_level': 'stress_risk'}, inplace=True)
    df['stress_risk'] = df['stress_risk'].apply(lambda x: 'yes' if x >= 60 else 'no')
    df['addiction_status'] = df['addiction_status'].map({'addicted': 1, 'not addicted': 0})
    df['stress_risk'] = df['stress_risk'].map({'yes': 1, 'no': 0})
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    X = df[features]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_logistic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def train_knn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return y_pred, y_score, acc, prec, rec, f1

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'Actual: {l}' for l in labels], columns=[f'Predicted: {l}' for l in labels])
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    return fig

def plot_roc_curve(y_test, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='darkorange', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} ROC Curve')
    ax.legend(loc="lower right")
    return fig

def plot_classification_heatmap(df, features):
    corr_matrix = df[features + ['addiction_status', 'stress_risk']].corr()
    corr_targets = corr_matrix[['addiction_status', 'stress_risk']].drop(['addiction_status', 'stress_risk'])
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(corr_targets, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation of Features with Target Variables')
    return fig

def plot_pairplot(df, target, features):
    df_plot = df[features + [target]].copy()
    df_plot[target] = df_plot[target].astype(str)
    fig = sns.pairplot(df_plot, hue=target)
    return fig
