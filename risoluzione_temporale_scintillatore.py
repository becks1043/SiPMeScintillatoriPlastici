#risoluzione_temporale_scintillatore
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from scipy.optimize import curve_fit
#quello che vogliamo fare è calcolarci le FWHM delle distribuzioni dei ritardi e capire coem variano al variare della
#tensione di overvoltage e al variare del CDF

#diverse tensioni di overvoltage del SiPM
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\risT_112.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\risT_114.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\risT_116.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\risT_118.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\risT_120.txt"

#diverse CDF del SiPM 
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_10.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_40.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_50.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_60.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_70.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_80.txt"
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\risoluzione_temporale\CFD_90.txt"

name =os.path.basename(file_path)    
 
time, count = [],[]
for i in range(len(file_path)):
    with open(file_path, "r") as file:
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


time = [x*1e9 for x in time ]
count_err = [np.sqrt(x) for x in count]
count_err = [1 if 0==x else x for x in count_err]

def gaussian(x, a, mu, sigma):
     return a * np.exp(-((x-mu)**2)/(2*(sigma**2)))

popt, pcov = curve_fit(gaussian, time , count, sigma= count_err, absolute_sigma= False)
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
plt.xlabel("ritardi [ns]")
plt.ylabel("conteggi [adm]")
plt.grid(which="both", linestyle="--")
plt.legend(title = f"$\sigma$ = {np.round(sigma*1000,2)}$\pm${np.round(np.sqrt(pcov[2,2])*1000,2)}[ps]")
plt.title(f"Risoluzione temporale {name}")
plt.show()


