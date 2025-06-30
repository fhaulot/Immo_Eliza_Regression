# Regression

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

temp = np.array([10, 15, 20, 25, 30]).reshape(-1, 1)
sales = np.array([20, 35, 50, 65, 80])

model = LinearRegression()
model.fit(temp, sales)
sales_pred = model.predict(temp)

plt.scatter(temp, sales)
plt.plot(temp, sales_pred, color='red')
plt.title('Ice Cream Sales vs. Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Sales')
plt.show()