# =========================================================
# INFERENCE + LATENCY TESTING
# =========================================================

import joblib
import time
import numpy as np

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("iris_model.pkl")

print("Model loaded successfully!")

# =========================================================
# SAMPLE INPUT
# =========================================================

sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

# =========================================================
# LATENCY MEASUREMENT
# =========================================================

start_time = time.time()

prediction = model.predict(sample_data)

end_time = time.time()

latency = (end_time - start_time) * 1000

# =========================================================
# OUTPUT
# =========================================================

print("\nPrediction:", prediction[0])

print(f"Latency: {latency:.4f} ms")