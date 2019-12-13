import matplotlib.pyplot as plt
import numpy as np
x=np.array(range(-50,50))
plt.plot(x,np.log(2.71828**1-x/(1+2.71828**(3-x)*(1+2.71828**(1-x)))),'-r',label = 'Bias')
plt.show()
