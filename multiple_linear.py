import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
# print(X_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# def predict_single_loop(x, w, b): 
#     """
#     single predict using linear regression
    
#     Args:
#       x (ndarray): Shape (n,) example with multiple features
#       w (ndarray): Shape (n,) model parameters    
#       b (scalar):  model parameter     
      
#     Returns:
#       p (scalar):  prediction
#     """
#     n = x.shape[0]
#     p = 0
#     print(n)
#     for i in range(n):
#         p_i = x[i] * w[i]  
#         p = p + p_i         
#     p = p + b                
#     return p

# x_vec = X_train[0,:]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# # make a prediction
# f_wb = predict_single_loop(x_vec, w_init, b_init)
# print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# def predict(x, w, b): 
#     """
#     single predict using linear regression
#     Args:
#       x (ndarray): Shape (n,) example with multiple features
#       w (ndarray): Shape (n,) model parameters   
#       b (scalar):             model parameter 
      
#     Returns:
#       p (scalar):  prediction
#     """
#     p = np.dot(x, w) + b     
#     return p 

# # get a row from our training data
# x_vec = X_train[0,:]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# # make a prediction
# f_wb = predict(x_vec,w_init, b_init)
# print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m) :
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(cost)

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

