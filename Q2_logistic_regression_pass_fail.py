
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data
study_hours = np.array([[1], [2], [3], [4], [5], [6]])
pass_fail = np.array([0, 0, 0, 1, 1, 1])

# Model
log_reg = LogisticRegression()
log_reg.fit(study_hours, pass_fail)

# Prediction
hours_range = np.linspace(0, 7, 100).reshape(-1, 1)
probabilities = log_reg.predict_proba(hours_range)[:, 1]

# Plot
plt.plot(hours_range, probabilities, color='green')
plt.scatter(study_hours, pass_fail, color='black', label='Actual Data')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Pass/Fail Prediction")
plt.grid(True)
plt.legend()
plt.show()
