#curvaIV_inversa_derivata

import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, fsolve

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\LED2.txt"
data = np.loadtxt(file_path)
name = os.path.basename(file_path)
base_name, ext = os.path.splitext(name)

#data set
x_data = data[: , 0] #volts V
y_data = data[: , 1] #current nA
y_data *= 1e-9

mask = y_data < 1e-6
x_filtered = np.array(x_data[mask])
y_filtered = np.array(np.log(y_data[mask]))
#y_err_filtered = np.array([0.02*x for x in y_filtered])
x_err_filtered = np.array([0.02*x for x in x_filtered])
y_err_filtered = np.full(np.shape(y_filtered)[0], 0.02)

def linear(x, a, b):
    return a*x + b
def exp(x,a,c):
    return a*np.exp(x) + c 

retta2 = (x_filtered >= 52)
retta1 = x_filtered <= 52

guess= None
popt1, pcov1 = curve_fit(linear, x_filtered[retta1], y_filtered[retta1], sigma = y_err_filtered[retta1], absolute_sigma = False)
popt2, pcov2 = curve_fit(linear, x_filtered[retta2], y_filtered[retta2], p0 = guess ,sigma = y_err_filtered[retta2], absolute_sigma = False)
#troviamo il pt d'intersezione
print(x_filtered[retta2])
def diff(x):
    return linear(x, *popt1) - linear(x, *popt2)

x0 = 52
x_intersect = fsolve(diff, x0)[0]
y_intersect = linear(x_intersect, *popt1)

x1 = np.linspace(0, max(x_filtered[retta1]), 1000)
x2 = np.linspace(0, max(x_filtered[retta2]), 1000)
x = np.linspace(0, 60, 1000)
#plot
plt.xlim(0, max(x_filtered))
plt.ylim(min(y_filtered) , max(y_filtered))
plt.xlabel("Pol inversa [V]")
plt.ylabel("ln(Corrente) [ln(A)]")
plt.plot(x_intersect, y_intersect, "^", color="r", label=f"intersezione a {np.trunc(x_intersect)} V",  zorder=5)
plt.errorbar(x_filtered, y_filtered, y_err_filtered, x_err_filtered, fmt=".", color="green", capsize=1)
plt.plot(x, linear(x, *popt1), label="fit lineare", color="orange")
plt.plot(x, linear(x, *popt2), label="fit lineare", color="blue")
plt.grid(True, linestyle= "--")
plt.title(f"Curva I-V di {base_name}")
plt.legend()
plt.show()