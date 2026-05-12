# =========================================================
# TRAINING SCRIPT
# =========================================================

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================================================
# LOAD DATASET
# =========================================================

wine = load_wine()

X = wine.data
y = wine.target

print("Dataset shape:", X.shape)

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================================================
# PIPELINE
# Preprocessing + Model
# =========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# =========================================================
# TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)

# =========================================================
# VALIDATION
# =========================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(pipeline, "wine_model.pkl")

print("\nModel saved as wine_model.pkl")