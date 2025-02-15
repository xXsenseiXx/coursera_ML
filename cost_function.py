import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
min_cost = 10000

def compute_cost(a, b, x, y) :
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = a * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost + cost_sum
    total_cost = (1/(2*m)) * cost_sum

    return total_cost


for i in range(0, 10000) :
    w = np.random.randint(0, 1000)
    b = np.random.randint(0, 1000)
    cost = compute_cost(w, b, x_train, y_train)
    if cost < min_cost :
        min_cost = cost
        best_w = w
        best_b = b

print(min_cost , best_w, best_b)

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    print (f_wb)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, best_w, best_b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Since you can't install `lab_utils_uni`, you can comment out or remove the line that uses it.
# plt_intuition(x_train, y_train)

# Alternatively, you can replace it with your own plotting function if needed.
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Training Data")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.show()