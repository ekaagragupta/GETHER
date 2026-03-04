import matplotlib.pyplot as plt

plt.plot(y_test, label="Actual AQI")
plt.plot(model.predict(X_test), label="Predicted AQI")

plt.legend()
plt.title("AQI Prediction")
plt.show()