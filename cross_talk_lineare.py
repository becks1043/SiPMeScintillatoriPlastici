import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

v_bd = 52.3 #v
x_data = [54, 55, 56, 57, 58]
y_data = [0.32, 1.9, 5.4, 7, 8.47]
x_data = [x - v_bd for x in x_data]

def linear(x, a , b):
    return x*a + b

popt, pcov= curve_fit(linear,x_data, y_data)
x= np.linspace(0, 10, 1000)
plt.scatter(x_data, y_data, color="red", cmap="1")
plt.plot(x, linear(x, *popt), color="blue")
plt.xlabel("overvoltage [V]")
plt.ylabel("percentuale [adm]")
plt.title(f"")
plt.xlim(0,10)
plt.ylim(0,60)
plt.grid()
plt.show()
