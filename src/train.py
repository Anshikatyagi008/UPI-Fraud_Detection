import pandas as pd
import numpy as np
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================
train_trans = pd.read_csv("data/train_transaction.csv")
train_id = pd.read_csv("data/train_identity.csv")

df = train_trans.merge(train_id, on="TransactionID", how="left")

df["txn_count"] = df.groupby("card1")["TransactionAmt"].transform("count")
df["avg_amt"] = df.groupby("card1")["TransactionAmt"].transform("mean")
df["amt_dev"] = df["TransactionAmt"] - df["avg_amt"]


features = [
    'TransactionAmt', 'ProductCD', 'card1', 'card4', 'card6',
    'addr1', 'dist1', 'P_emaildomain',
    'txn_count', 'avg_amt', 'amt_dev'
]

df = df[features + ['isFraud']]

# Fill missing
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(-999)

cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("unknown")

# Encode
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop("isFraud", axis=1)
y = df["isFraud"]


X["amt_log"] = np.log1p(X["TransactionAmt"])
X["amt_squared"] = X["TransactionAmt"] ** 2
X["amt_inverse"] = 1 / (X["TransactionAmt"] + 1)
X["amt_sqrt"] = np.sqrt(X["TransactionAmt"])
X["is_high_amt"] = (X["TransactionAmt"] > 20000).astype(int)
X["is_low_amt"] = (X["TransactionAmt"] < 100).astype(int)
X["amt_range"] = pd.cut(X["TransactionAmt"], bins=5, labels=False)
# =========================
# FIX RANDOMNESS (IMPORTANT)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = XGBClassifier(
    n_estimators=900,
    max_depth=7,
    learning_rate=0.02,
    scale_pos_weight=12,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='auc'
)


model.fit(X_train, y_train)

# =========================
# AUC
# =========================
y_scores_test = model.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
roc_auc = auc(fpr_test, tpr_test)

print("🔥 AUC SCORE:", roc_auc)

# =========================
# SAVE MODEL
# =========================
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ Model trained successfully")