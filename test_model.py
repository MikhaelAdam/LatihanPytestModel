# =========================================================
# PYTEST UNIT TESTING
# =========================================================

import pytest
import joblib
import numpy as np

# =========================================================
# FIXTURE
# Load model once
# =========================================================

@pytest.fixture
def model():
    return joblib.load("wine_model.pkl")


# =========================================================
# TEST 1
# Check model loaded
# =========================================================

def test_model_loaded(model):
    assert model is not None


# =========================================================
# TEST 2
# Check prediction shape
# =========================================================

def test_prediction_shape(model):

    sample_data = np.array([
        [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
    ])

    prediction = model.predict(sample_data)

    assert len(prediction) == 1


# =========================================================
# TEST 3
# Known prediction
# Wine cultivar class 0
# =========================================================

def test_known_prediction(model):

    sample_data = np.array([
        [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
    ])

    prediction = model.predict(sample_data)

    # Expected Wine cultivar = class 0
    assert prediction[0] == 0


# =========================================================
# TEST 4
# Probabilities sum to 1
# =========================================================

def test_probability_sum(model):

    sample_data = np.array([
        [12.86, 1.35, 2.32, 18.0, 122.0, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630.0]
    ])

    probabilities = model.predict_proba(sample_data)

    total_prob = probabilities[0].sum()

    assert total_prob == pytest.approx(1.0, abs=1e-6)


# =========================================================
# TEST 5
# Parameterized Testing
# =========================================================

@pytest.mark.parametrize(
    "sample_data, expected_class",
    [
        ([[14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]], 0),
        ([[12.37, 0.94, 1.36, 10.6,  88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82,  520.0]], 1),
        ([[12.86, 1.35, 2.32, 18.0, 122.0, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29,  630.0]], 2),
    ]
)

def test_multiple_predictions(model, sample_data, expected_class):

    prediction = model.predict(sample_data)

    assert prediction[0] == expected_class