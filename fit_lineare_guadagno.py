#fit_lineare_guadagno
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

x_data = np.array([59., 58., 57., 56., 55., 54.])
y_data = np.array([4633395.93, 3994060.11, 3313016.88, 2602567.84, 1883948.75, 1184615.88 ]) #guadagni
y_err = np.array([11459.16, 5547.14, 9339.43, 3738.55, 7278.66, 13354.15])

#y_data *= 1e-6
#y_err *= 1e-6
def linear(x, a, b):
    return a*x + b

popt, pcov = curve_fit(linear, x_data, y_data)
v_bd = np.abs(popt[1]/popt[0])
v_bd_error = v_bd*np.sqrt((np.sqrt(pcov[0,0])/popt[0])**2+(np.sqrt(pcov[1,1])/popt[1])**2 -  2*pcov[0,1]/(popt[0]*popt[1]))

#guadagno al break down: intercetta della retta con l'asse zero
print(f"------\nV_bd={v_bd}+-{v_bd_error}")

#plot
x = np.linspace(0, 60, 10000)
plt.errorbar(x_data ,y_data ,yerr=y_err, color='green', fmt=".", capsize=2)
plt.plot(x, linear(x, *popt), label = "fit",color="red")
plt.plot(x, linear(x,0,0), "--", color="black")
plt.xlabel("Volts [V]")
plt.ylabel("Guadagno [adm]")
plt.title(f"linearità")
plt.legend(title= f"V_bd={np.round(v_bd,2)}+-{np.round(v_bd_error,2)}")
plt.xlim(52, 60)
plt.ylim(-1, 6e6)
plt.grid()
plt.show()

#si rifà il fit ma shiftato con la tensione di overvoltage
x_data1 = x_data - v_bd
x1 = np.linspace(0, 8, 1000)

popt1, pcov1 = curve_fit(linear, x_data1, y_data)

plt.errorbar(x_data1, y_data ,yerr=y_err, color='green', fmt=".", capsize=2)
plt.plot(x1, linear(x1,*popt1), label = "fit",color="red")
plt.plot(x1, linear(x1,0,0), "--", color="black")
plt.xlabel("tensione di over voltage [V]")
plt.ylabel("Guadagno [adm]")
plt.title(f"Guadagno per un pixel pitch di 50 micron")
plt.xlim(0, 8)
plt.ylim(-1, 6e6)
plt.grid()
plt.show()

#capacità della microcella deve rimanere costante se il guadagno è lineare
q = 1.6e-19  #carica dell'elettrone
Q_tot = y_data*q
Q_tot *= 1e15
c = Q_tot / x_data1 #capacità della microcella

print(c)
popt2, pcov2 = curve_fit(linear, x_data1, Q_tot)
plt.errorbar(x_data1, Q_tot ,yerr= y_err*q, color='green', fmt=".", capsize=2)
plt.plot(x1, linear(x1,*popt2), label = "fit",color="red")
#plt.plot(x, linear(x,0,0), "--", color="black")
plt.xlabel("tensione di over voltage [V]")
plt.ylabel("Carica totale [fC]")
plt.title(f"Capacità della microcella")
plt.legend(title=f"capacità={np.round(popt2[0],4)}+-{np.round(np.sqrt(pcov2[0,0]),4)}fF")
#plt.xlim(0, 8)
#plt.ylim(-1, 6e6)
plt.grid()
plt.show()

