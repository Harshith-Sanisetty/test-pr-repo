
import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow.keras.datasets import mnist

# Load data
(X_train, _), (_, _) = mnist.load_data()

# Preprocess
X_flat = X_train.reshape(-1, 784)  # Flatten each image
avg_brightness = X_flat.mean(axis=1)

# Model
lin_reg_mnist = LinearRegression()
lin_reg_mnist.fit(X_flat, avg_brightness)

# Predict and display result for one image
pred_brightness = lin_reg_mnist.predict(X_flat[:5])
print("Predicted Brightness (first 5 images):", pred_brightness)
