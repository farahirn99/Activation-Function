import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-10,10)
y=[]
for i in range (x.shape[0]):
    y.append((1/(1+np.exp(-x[i]))))
plt.plot(x,y,c="blue")
plt.title("sigmoid")
plt.show()
print(sigmoid(2))
