#Rumore correlato
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import os 

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\Discriminatore56V.txt"
data = np.loadtxt(file_path)
name = os.path.basename(file_path)

#data set
x_data = data[: , 0] #volts mV
y_data = data[: , 1] #conteggi
x_data *= 1e-3

#identificazione degl outliner
mean = np.mean(y_data)
st_dev = np.std(y_data)

trashold = 1
lower_bound = mean - trashold*st_dev
upper_bound = mean + trashold*st_dev

mask1 = (y_data < lower_bound) | (y_data> upper_bound )
mask2 = (y_data>= lower_bound ) & (y_data <= upper_bound  ) & (y_data > 1e-9)

x_filtered = x_data[mask2]
y_filtered = y_data[mask2]
x_exluded = x_data[mask1]
y_excluded = y_data[mask1]
#probabilità di cross talk
x_filtered = list(x_filtered)
y_filtered = list(y_filtered)
#57V
"""
a = 0.168
b = 0.189
c = 0.215
"""
a = 0.128
b = 0.14
c = 0.200

a1 = [np.abs(x-a) for x in x_filtered ]
b1 = [np.abs(x-b) for x in x_filtered ]
c1 = [np.abs(x-c) for x in x_filtered ]

step_1 = min(a1)
step1 = a1.index(step_1)
step_2 = min(b1)
step2 = b1.index(step_2)
step_3 = min(c1)
step3 = c1.index(step_3)

N1 = y_filtered[:step1]
N2 = y_filtered[step2:step3]

n1 = np.mean(N1)
n2 = np.mean(N2)
print(n1, n2)
P_ct = n2/n1
print(f"-------\nprobabilità di cross talk {P_ct}")
plt.scatter(x_filtered, y_filtered, color="blue",marker=".")
plt.scatter(x_exluded, y_excluded, color="red",label = "dati esclusi", marker="*")
plt.yscale("log")
plt.grid(which="both", linestyle="--")
plt.xlabel("treshhold [mV]")
plt.ylabel("Conteggi [adm]")
plt.legend(title=f"Prabibiltà di cross talk = {np.round(P_ct*100,2)}%")
plt.title(f"{name}")

plt.show()