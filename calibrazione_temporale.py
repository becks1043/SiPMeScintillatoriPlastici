#calibrazione_temporale
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from scipy.optimize import curve_fit
#sulle x metto i ritardi dei tempi mentre sulle y le posizioni della sorgente
y_data = np.array([2., 6., 10., 14., 18., 22., 26., 30., 34., 38., 42.]) #posizione partendo da sx [cm]
y_err = np.array([0.1 for x in y_data]) #cm
x_data = np.array([-2.8, -2.31, -1.75, -1.19, -0.63, -0.1, 0.45, 1.04, 1.62, 2.16, 2.67])#ns
x_err = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 ]) #come errore prendo la sigma della distribuzione

guess = [None]
def gaussian(x, a, mu, sigma):
     return a * np.exp(-((x-mu)**2)/(2*(sigma**2)))

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\calibrazione_barra\sx_34cm.txt"

name =os.path.basename(file_path)    
class DataReader:
     def __init__(self, file_path):
          self.file_path = file_path
     def data(self):
        time, count = [],[]
    
        with open(self.file_path, "r") as file:
            reader = csv.reader(file)
            next(reader)
            for riga in reader:
                try:
                    if riga[0]:
                        time.append(float(riga[0]))
                        count.append(float(riga[1]))
                except ValueError:
                        #righe_str.append(numero)
                        continue
        return time, count
    

time, count = DataReader(file_path).data()
#print(time, count)

time = [x*1e9 for x in time ]
count_err = [np.sqrt(x) for x in count]
count_err = [1 if 0==x else x for x in count_err]

def gaussian(x, a, mu, sigma):
     return a * np.exp(-((x-mu)**2)/(2*(sigma**2)))

guess= [402.5, -1.81, 0.4] #per 10cm
guess = None
popt, pcov = curve_fit(gaussian, time , count, p0=guess, sigma= count_err, absolute_sigma= False)
a, mu, sigma = popt

print(f"-----")
print(f"sigma = {sigma}+-{np.sqrt(pcov[2,2])}")
print(f"media = {mu}+-{np.sqrt(pcov[1,1])}")
y_fit_th = gaussian(time, *popt) - count

chi_square = np.sum((y_fit_th /count_err)**2) #va diviso per gli errori 

dof = len(time) - len(popt)
chi_norm = chi_square / dof
print(f"chi quadro {chi_square}/{dof}\nchi norm {chi_norm}")

x = np.linspace(min(time), max(time), 100000)
plt.scatter(time, count, color = "blue", label="dati", marker=".")
plt.plot(x, gaussian(x, *popt), color="green", label="fit")
plt.ylabel("ritardi [ns]")
plt.xlabel("conteggi [adm]")
plt.grid(which="both", linestyle="--")
plt.legend(title = f"$\mu$ = {np.round(mu,2)}$\pm${np.round(np.sqrt(pcov[1,1]),3)}[ns]")
plt.title(f"Risoluzione temporale {name}")
plt.show()

#plot della velocità della luce nella barra

def linear(x,a, b):
     return a*x + b

popt1, pcov1 = curve_fit(linear, x_data, y_data, sigma= y_data, absolute_sigma= False)

for i in range(3):
    total_error= np.sqrt(y_err**2 + (popt[0]*x_err)**2)
    popt2, pcov2 = curve_fit(linear, x_data, y_data, sigma=total_error)
    chi_square1 = np.sum(((linear(x_data, *popt1) - y_data) /total_error[i])**2) #va diviso per gli errori 

dof1 = len(x_data) - len(popt1)
chi_norm1 = chi_square1 / dof1
print(f"chi quadro {chi_square1}/{dof1}\nchi norm {chi_norm1}")

x = np.linspace(min(x_data), max(x_data), 10000)
plt.errorbar(x_data, y_data, y_err, x_err, color = "blue", fmt=".", capsize= 2)
plt.plot(x, linear(x, *popt1), color="green")
plt.xlabel("ritardi [ns]")
plt.ylabel("posizione della sorgente da sx a dx [cm]")
plt.grid(which="both", linestyle="--")
plt.legend(title = f"$\chi^2$ = {np.round(chi_square1,2)}/ {dof1}\nv =({np.round(popt1[0]*2,2)}$\pm${np.round(np.sqrt(pcov1[0,0]),2)})x$10^7$[m/s]")
plt.title(f"Velocità della luce nella sbarra")
plt.show()


