import matplotlib.pyplot as plt
import numpy as np
x_train = [0, 2, 3, 5]
y_train = [1, 4, 9, 16]
x_test  = [1, 4]
y_test  = [3, 12]

def d1(x):
    return -0.1923+3.0769*x
def d2(x):
    return 0.80769+1.41*x+0.333*x**2
def d3(x):
    return 0.99999+(-2.833)*x+2.833*x**2+(-0.333)*x**3
def d4(x):
    return 0.99999+(-0.1396)*x+0.04986*x**2+0.5645*x**3+(-0.0897)*x**4
# y1=1;y2=4;y3=9;y4=16
#test point (1,3)(4,12)
def variance(i1,i2,i3,i4):
    return (1-i1)**2+(4-i2)**2+(9-i3)**2+(16-i4)**2

def bias(i1,i2):
    return (3-i1)**2+(12-i2)**2

# print(d4(0.5))

print(bias(d4(1),d4(4)))
print(variance(d4(0),d4(2),d4(3),d4(5)))
print("total error=",bias(d4(1),d4(4))+variance(d4(0),d4(2),d4(3),d4(5)))
print("train err=",variance(d4(0),d4(2),d4(3),d4(5))/4)
print("test err=",bias(d4(1),d4(4))/2)

d=np.array([1,2,3,4])
arrbias=np.array([0.0266,0.25219,8.2787,8.388])
arrvariance=np.array([5.92307,1.9232,0.00127,0.00238])
arrtotalerr=np.array([5.94968125,2.1754,8.28,8.3904])
arrtrainerr=np.array([1.4808,0.4808,0.000319,0.000596])
arrtesterr=np.array([0.0133,0.1261,4.139,4.194])


plt.plot(d,arrbias,'-r',label = 'Variance')
plt.plot(d,arrvariance,'-g',label = 'Bias')
plt.plot(d,arrtotalerr,'-b',label = 'Total err')
plt.plot(d,arrtrainerr,'-y',label = 'Training err')
plt.plot(d,arrtesterr,'-c',label = 'Test err')

plt.legend()
plt.show()
