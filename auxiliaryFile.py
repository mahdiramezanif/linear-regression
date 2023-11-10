import numpy as np
import matplotlib.pyplot as plt
import math
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')

x = np.array([2, 1.2, 3])
y = np.array([2000, 1500, 2800])
number_of_examples = x.shape[0]


def cost(x2, y2, w2, b2, number_of_examples2):
    cost_sum = 0

    for i in range(number_of_examples2):
        fwb = w2 * x2[i] + b2
        cost_sum += (fwb - y2[i]) ** 2
    total_cost = (1 / (2 * number_of_examples2)) * cost_sum

    return total_cost


def gradient(x3, y3, w3, b3, number_of_examples3):
    dj_dw = dj_db = 0

    for i in range(number_of_examples3):
        fwb = w3 * x3[i] + b3
        dj_db += fwb - y3[i]
        dj_dw += (fwb - y3[i]) * x3[i]

    dj_dw /= number_of_examples3
    dj_db /= number_of_examples3

    return dj_dw, dj_db


def gradient_descent(x4, y4, w4, b4, alpha, num_iteration):
    j_history = []
    p_history = []

    for i in range(num_iteration):
        dj_dw, dj_db = gradient(x4, y4, w4, b4, number_of_examples)

        b4 -= alpha * dj_db
        w4 -= alpha * dj_dw

        if i < 100000:  # prevent resource exhaustion
            j_history.append(cost(x4, y4, w4, b4, number_of_examples))
            p_history.append([w4, b4])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iteration / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w4: 0.3e}, b:{b4: 0.5e}")

    return w4, b4


w_final, b_final = gradient_descent(x, y, 0, 0, 0.005, 10000)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
