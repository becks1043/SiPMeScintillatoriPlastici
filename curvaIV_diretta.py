import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\diretta_meno1800mV_step01_2s.txt"
data = np.loadtxt(file_path)
name = os.path.basename(file_path)

#data set
x_data = data[: , 0] #volts V
x_data = - x_data
y_data = data[: , 1] #current nA
y_data = - y_data
y_data *= 1e-6
y_err = [0.02*x for x in y_data]
x_err = [0.02*x for x in x_data]

x_data = np.array(x_data)
y_data = np.array(y_data)
x_err = np.array(x_err)
y_err = np.array(y_err)
 
def linear(x,a,b):
    return a*x + b

def linear_derivate(x):
    return a_fit

popt, pcov = curve_fit(linear, x_data[7:] , y_data[7:], p0=None)
a_fit, b_fit = popt

print(f"-----\n {popt}")
y_fit_th = linear(x_data[7:], *popt)
total_error= np.sqrt(y_err[7:]**2 + (linear_derivate(x_data[7:])*x_err[7:])**2)

for i in range(len(y_data[7:])):
    chi_square = np.sum((y_fit_th /total_error[i])**2) #va diviso per gli errori 

dof = len(x_data[7:]) - len(popt)
chi_norm = chi_square / dof
print(f"chi quadro {chi_square}\nchi norm {chi_norm}")

#plot
plt.errorbar(x_data , y_data, yerr= y_err,xerr= x_err , fmt=".",color='blue',capsize=1)
plt.plot(x_data, linear(x_data,*popt), color="red")
plt.xlabel("Pol diretta [V]")
plt.ylabel("Corrente [mA]")
plt.title(f"Curva I-V {name}")
plt.xlim(min(x_data), max(x_data))
plt.ylim(-0.1,)
plt.legend(title =f"N/R_q = {np.round(a_fit,2)} +- {np.round(np.sqrt(pcov[0,0]), 2)}")
plt.grid(which="both", linestyle="--")
plt.show()