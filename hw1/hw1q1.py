import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = np.array([0, 2, 3, 5]).reshape((-1, 1))
# print(x)
y = np.array([1, 4, 9, 16])
xtest=np.array([1, 3])
ytest=np.array([4, 12])

# lin = LinearRegression()
# lin.fit(x, y)
x1=0
y1=7.5
polyd1 = PolynomialFeatures(degree=1, include_bias=False)
polyd2 = PolynomialFeatures(degree=2, include_bias=False)
polyd3 = PolynomialFeatures(degree=3, include_bias=False)
polyd4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly1 = polyd1.fit_transform(x)
X_poly2 = polyd2.fit_transform(x)
X_poly3 = polyd3.fit_transform(x)
X_poly4 = polyd4.fit_transform(x)


# transformer.fit(X_poly, y)
lin1 = LinearRegression()
lin2 = LinearRegression()
lin3 = LinearRegression()
lin4 = LinearRegression()
lin1.fit(X_poly1, y)
lin2.fit(X_poly2, y)
lin3.fit(X_poly3, y)
lin4.fit(X_poly4, y)

# model = LinearRegression().fit(x_, y)
# r_sq = model.score(x_, y)
# print('coefficient of determination:', r_sq)
print('d=1 coefficients:', lin1.coef_)
print('d=1 a0:', lin1.intercept_)
print('d=2 coefficients:', lin2.coef_)
print('d=2 a0:', lin2.intercept_)
print('d=3 coefficients:', lin3.coef_)
print('d=3 a0:', lin3.intercept_)
print('d=4 coefficients:', lin4.coef_)
print('d=4 a0:', lin4.intercept_)

xplot = np.arange(1,7)
# xplot=np.array([0,1,2,3,4,5,6,7,8,9,10]
# print(xplot)
# print('predict:',lin1.predict(polyd1.fit_transform(x)))
plt.scatter(x, y, color = 'blue',label ='trainning data')
plt.scatter(xtest, ytest, color = 'red',label ='test data')

plt.plot([0,5],[7.5,7.5],label ='d=0')
plt.plot(x, lin1.predict(polyd1.fit_transform(x)), '-r',label ='d=1')#red
plt.plot(x, lin2.predict(polyd2.fit_transform(x)), ':c',label ='d=2')#cyan
plt.plot(xplot, 0.99999+(-2.833)*xplot+2.833*xplot**2+(-0.333)*xplot**3, '-g',label ='d=3')
# plt.plot(x, lin3.predict(polyd3.fit_transform(x)), '-g',label ='d=3')#green
# plt.plot(x, lin4.predict(polyd4.fit_transform(x)), '-y',label ='d=4')#yellow
plt.plot(xplot, 0.99999+(-0.1396)*xplot+0.04986*xplot**2+0.5645*xplot**3+(-0.0897)*xplot**4, '-y',label ='d=4')
# plt.plot(x, lin2.predict(PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)), color = 'yellow')
plt.legend()
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# TO GET predict y value
