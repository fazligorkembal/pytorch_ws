import numpy as np

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

print("x:")
print(x)
print("y:")
print(y)

x_b = np.c_[np.ones((100, 1)), x]
print("x_b:")
print(x_b)

theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print("theta_best:")
print(theta_best)

x_new = np.array([[0], [2]])
print("x_new:")
print(x_new)
x_new_b = np.c_[np.ones((2, 1)), x_new]
print("x_new_b:")
print(x_new_b)
y_predict = x_new_b.dot(theta_best)
print("y_predict:")
print(y_predict)

import matplotlib.pyplot as plt
plt.plot(x_new, y_predict, "r-")
plt.plot(x, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()