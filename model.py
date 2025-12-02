from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Smooth ROC curve functions
def smooth_roc(fpr, tpr, points=300):
    fpr_new = np.linspace(0, 1, points)
    tpr_new = np.interp(fpr_new, fpr, tpr)
    return fpr_new, tpr_new

def plot_roc_curves_smooth(y_test, prob_lf, prob_rf, prob_xgb, vegas_prob):
    plt.figure(figsize=(10, 8))

    models = [
        ("Logistic Regression", prob_lf),
        ("Random Forest", prob_rf),
        ("XGBoost", prob_xgb),
        ("Vegas Odds Baseline", vegas_prob)
    ]

    colors = ["#000000", "#ff0000", "#00ff00", "#0000ff"]
    ls = ["solid", "solid", "solid", "solid"]

    for (label, prob), color, style in zip(models, colors, ls):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = auc(fpr, tpr)

        # Smooth the curve
        fpr_s, tpr_s = smooth_roc(fpr, tpr)

        plt.plot(
            fpr_s, tpr_s,
            label=f"{label} (AUC = {auc_val:.3f})",
            linewidth=2.2,
            color=color,
            linestyle=style
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess")

    plt.title("Smoothed ROC Curves for All Models", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.show()


def ml_to_prob(ml):
    if pd.isna(ml):
        return 0.5
    ml = float(ml)
    # Favorite vs underdog
    if ml < 0:   
        return -ml / (-ml + 100)
    else:        
        return 100 / (ml + 100)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Away (0)", "Pred Home (1)"],
        yticklabels=["True Away (0)", "True Home (1)"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

df = pd.read_csv("data/mlb-2025-with-starter-stats.csv")

# One-hot encode teams
team_dummies = pd.get_dummies(df[["Home", "Away"]], prefix=["HomeTeam", "AwayTeam"])
df = pd.concat([df, team_dummies], axis=1)

team_features = list(team_dummies.columns)

# Pitcher values only
feature_cols = [
    "Home_ERA_before", "Home_FIP_before", "Home_WHIP_before",
    "Home_K9_before", "Home_BB9_before", "Home_HR9_before",
    "Home_IP_per_start_before",
    "Away_ERA_before", "Away_FIP_before", "Away_WHIP_before",
    "Away_K9_before", "Away_BB9_before", "Away_HR9_before",
    "Away_IP_per_start_before",

    "Home_ERA_last3_before", "Home_WHIP_last3_before", "Home_K9_last3_before",
    "Home_BB9_last3_before", "Home_HR9_last3_before", "Home_IP_last3_before",
    "Away_ERA_last3_before", "Away_WHIP_last3_before", "Away_K9_last3_before",
    "Away_BB9_last3_before", "Away_HR9_last3_before", "Away_IP_last3_before",
] + team_features

# Fix ML values for Vegas baseline only
for col in ["Home ML", "Away ML"]:
    df[col] = df[col].replace("-", np.nan)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Train/Test Split Data
cutoff_date = "2025-09-01"

train_mask = df["Date_Start"] < cutoff_date
test_mask  = df["Date_Start"] >= cutoff_date

X_train = df.loc[train_mask, feature_cols].fillna(0)
y_train = df.loc[train_mask, "Home_Win"]

X_test = df.loc[test_mask, feature_cols].fillna(0)
y_test = df.loc[test_mask, "Home_Win"]

# Logistic Regression Model
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

pred_lf = clf.predict(X_test)
prob_lf = clf.predict_proba(X_test)[:, 1]

print("LR Accuracy:", accuracy_score(y_test, pred_lf))
print("LR AUC:", roc_auc_score(y_test, prob_lf))

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]

print("RF Accuracy:", accuracy_score(y_test, pred_rf))
print("RF AUC:", roc_auc_score(y_test, prob_rf))

#XGBoost Model
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)
prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("XGBoost Accuracy:", accuracy_score(y_test, pred_xgb))
print("XGBoost AUC:", roc_auc_score(y_test, prob_xgb))

# Vegas Heuristic Model
home_prob_raw = df.loc[test_mask, "Home ML"].apply(ml_to_prob)
away_prob_raw = df.loc[test_mask, "Away ML"].apply(ml_to_prob)
total = home_prob_raw + away_prob_raw
home_prob = home_prob_raw / total

vegas_pred = (home_prob_raw > away_prob_raw).astype(int)

print("Vegas Accuracy:", accuracy_score(y_test, vegas_pred))
print("Vegas AUC:", roc_auc_score(y_test, home_prob))

# Confusion Matrixes
plot_confusion_matrix(y_test, pred_lf, title="Logistic Regression")
plot_confusion_matrix(y_test, pred_rf, title="Random Forest")
plot_confusion_matrix(y_test, pred_xgb, title="XGBoost")

# Plot ROC
plot_roc_curves_smooth(
    y_test,
    prob_lf,
    prob_rf,
    prob_xgb,
    home_prob)
