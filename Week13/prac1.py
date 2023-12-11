# [기말 범위] [데이터분석및활용]선형회귀기본실습
import matplotlib.pyplot as plt
import numpy as np
import random

def gen_data(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)

    for i in range(0, numPoints):
        x[i][0] = 1 
        x[i][1] = i 
        y[i] = (i+bias) + random.uniform(0, 1) * variance 
    return x, y

def gradient_descent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    theta_list = []                               
    cost_list = []                                
    for i in range(0, numIterations):             
        hypothesis = np.dot(x, theta)            
        loss = hypothesis - y                     
        cost = np.sum(loss ** 2) / (2 * m)        
        gradient = np.dot(xTrans, loss) / m       
        theta = theta - alpha * gradient          
        if i % 250 == 0:                          
            theta_list.append(theta)
        cost_list.append(cost)
    return theta, np.array(theta_list), cost_list 

x, y = gen_data(100, 25, 10)

m, n = np.shape(x)   
numIterations = 20000
alpha = 0.0005       
theta = np.ones(n)   

print(m, n)

theta, theta_list, cost_list = gradient_descent(x, y, theta, alpha, m, numIterations)
y_predict_step= np.dot(x, theta_list.transpose())


plt.subplot(1, 3, 1)
plt.plot(x[:, 1], y, "ro")    
plt.title('data scatter')
plt.subplot(1, 3, 2)
plt.title('check theta')
for i in range (0, 80, 8):
    plt.plot(x[:,1], y_predict_step[:,i], label='Line %02d'%i)

plt.subplot(1, 3, 3)
plt.title('check cost')
iterations = range(len(cost_list))
plt.plot(iterations, cost_list, "o-")
plt.show()