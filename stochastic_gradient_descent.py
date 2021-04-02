import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Create given vectors: $x1, x2, y$
#Initialize learning rate: $n$
#Initialize amount of batches: $batchAmount$
x1 = np.array([4,2,1,3,1,6])
x2 = np.array([1,8,0,2,4,7])
y = np.array([2,-14,1,-1,-7,-8])
n = 0.05
batchAmount = 6

#Create vectors that should be computed: $ y\_hat, loss, y\_y\_hat, w1, w2, b $. Weights and bias are initially assigned.
y_hat = np.array([])
loss = np.array([])
y_y_hat = np.array([])
w1 = np.array([-0.017])
w2 = np.array([-0.048])
b = np.array([0])

#Loop 6 times to find the values and update $w1, w2$ and $b$
for i in range(batchAmount):
  #Compute the values
  y_hat = np.append(y_hat, w1[i]*x1[i] + w2[i]*x2[i] + b[i])
  y_y_hat = np.append(y_y_hat, y_hat[i] - y[i] )
  loss = np.append(loss, y_y_hat[i]*y_y_hat[i])

  #Time to update weights and bias
  #Check if this is the last update
  if i < (batchAmount - 1) :
    #Update the bias
    b = np.append(b, b[i] - n*2*y_y_hat[i])
    #Update the weights
    w1 = np.append(w1, w1[i] - n*2*y_y_hat[i]*x1[i])
    w2 = np.append(w2, w2[i] - n*2*y_y_hat[i]*x2[i])

#Create a datatable to insert all the info
d = {'x1': x1, 'x2': x2, 'y': y, 'y^': y_hat, 'loss': loss, 'y^-y': y_y_hat, 'w1': w1, 'w2': w2, 'b': b}
df = pd.DataFrame(data=d)
print(df)

#Graph the Loss vs. Batch number
plt.plot(range(batchAmount), loss)
plt.suptitle('Loss vs. Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()
