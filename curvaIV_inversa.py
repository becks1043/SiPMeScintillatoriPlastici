import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\inversa_60V_step05_1s.txt"
data = np.loadtxt(file_path)
name = os.path.basename(file_path)

#data set
x_data = data[: , 0] #volts V
y_data = data[: , 1] #current nA
y_data *= 1e-9

#identificazione degl outliner
mean = np.mean(y_data)
st_dev = np.std(y_data)

trashold = 3
lower_bound = mean - trashold*st_dev
upper_bound = mean + trashold*st_dev

mask1 = (y_data < lower_bound) | (y_data> upper_bound )
mask2 = (y_data>= lower_bound ) & (y_data <= upper_bound  ) & (y_data > 1e-9)

x_filtered = x_data[mask2]
y_filtered = y_data[mask2]

print(f'La shape del tensore dei dati è {np.shape(data)}\nLa shape del tensore di x_data è {np.shape(x_data)}')
#amperometro molto rumoroso e scalibrato
#non inseriamo l'errore perchè è una stima
#volendo inserire l'errore possiamo mettere il 2%
#sicuramente l'errore sulle ddp è maggiore
 
#plt.subplot(2,1,1)
plt.scatter(x_filtered , y_filtered, color='green')
plt.xlabel("Pol inversa [V]")
plt.ylabel("Corrente [A]")
plt.title(f"Curva I-V {name}")
plt.xlim(min(x_data), max(x_data))
plt.ylim()
plt.yscale("log")


plt.grid(which="both", linestyle="--")
plt.show()

