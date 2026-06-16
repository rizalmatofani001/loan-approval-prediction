import pandas as pd
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = Path("loan_approval_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_loan_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

selected_features = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score"
]
categorical_features = ["education", "self_employed"]
numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score"
]
target_col = "loan_status"

metadata = {
    "selected_features": selected_features,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "target_col": target_col,
    "target_mapping": {
        "Approved": 1,
        "Rejected": 0
    }
}

if not DATA_PATH.exists():
    raise FileNotFoundError("File loan_approval_dataset.csv tidak ditemukan. Letakkan file CSV satu folder dengan train_model.py.")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

required_cols = selected_features + [target_col]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan pada dataset: {missing_cols}")

df_model = df[required_cols].copy()
df_model[target_col] = df_model[target_col].astype(str).str.strip().map({
    "Approved": 1,
    "Rejected": 0,
    "approved": 1,
    "rejected": 0
})
df_model = df_model.dropna()

X = df_model[selected_features]
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf_model)
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_proba)
    ]
})

cat_names = model.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names = list(cat_names) + numeric_features
importances = model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

metrics_df.to_csv(METRICS_PATH, index=False)
feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

print("Model berhasil dibuat ulang.")
print(metrics_df)
