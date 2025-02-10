import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

buys_df = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\merlin\Neural Network\v3\buy_metrics_1mo.csv")
sells_df = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\merlin\Neural Network\v3\sell_metrics_1mo.csv")

buys_X = buys_df.drop(["Ticker", "Growth", "Date"], axis=1)
buys_y = np.ones(buys_X.shape[0])
sells_X = sells_df.drop(["Ticker", "Growth", "Date"], axis=1)
sells_y = np.zeros(sells_X.shape[0])
X = pd.concat([buys_X, sells_X]).values
y = np.concatenate([buys_y, sells_y])

sample_weights = compute_sample_weight(class_weight='balanced', y=y)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective(trial):
    param = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.1, 10.0)
    }
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **param
    )
    
    model.fit(
        X_train_scaled, y_train, 
        sample_weight=w_train, 
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return roc_auc

study = optuna.create_study(direction='maximize', sampler=TPESampler())

study.optimize(objective, n_trials=50)

best_params = study.best_params
best_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    **best_params
)

best_model.fit(
    X_train_scaled, y_train, 
    sample_weight=w_train, 
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False
)

y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

best_model.save_model(r"C:\Users\arnav\OneDrive\Documents\Merlin\Neural Network\model_xgb_1mo.json")

print("Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label="XGBoost")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

buys_df = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\merlin\Neural Network\v3\buy_metrics_1wk.csv")
sells_df = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\merlin\Neural Network\v3\sell_metrics_1wk.csv")

buys_X = buys_df.drop(["Ticker", "Growth", "Date"], axis=1)
buys_y = np.ones(buys_X.shape[0])
sells_X = sells_df.drop(["Ticker", "Growth", "Date"], axis=1)
sells_y = np.zeros(sells_X.shape[0])
X = pd.concat([buys_X, sells_X]).values
y = np.concatenate([buys_y, sells_y])

sample_weights = compute_sample_weight(class_weight='balanced', y=y)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective(trial):
    param = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.1, 10.0)
    }
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **param
    )
    
    model.fit(
        X_train_scaled, y_train, 
        sample_weight=w_train, 
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return roc_auc

study = optuna.create_study(direction='maximize', sampler=TPESampler())

study.optimize(objective, n_trials=50)

best_params = study.best_params
best_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    **best_params
)

best_model.fit(
    X_train_scaled, y_train, 
    sample_weight=w_train, 
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False
)

y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

best_model.save_model(r"C:\Users\arnav\OneDrive\Documents\Merlin\Neural Network\model_xgb_1wk.json")

print("Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label="XGBoost")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()