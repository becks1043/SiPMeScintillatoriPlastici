import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import csv

path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\F1--Guadagno56V--00000.txt"
name =os.path.basename(path)

tempo = [] #s
y_data = []
data = []
with open(path, "r") as file:
    reader = csv.reader(file)
    for _ in range (5):
        next(reader)

    for row in reader:
        if row:
            tempi, ampiezze = map(float, row)
            tempo.append(tempi)
            y_data.append(ampiezze)

tempo = [x*1e9 for x in tempo] #nVs
#x_data = np.linspace(1, 1000, 1000)

time = []
for i in range(len(tempo)):
    if tempo[i]>0:
        time.append(tempo[i])
    else:
        continue

start = len(time)
a = len(y_data)
y_data = y_data[a- start:]
x_data = time
#plot dei dati

plt.plot(x_data, y_data, color="blue")
plt.xlabel("area (nV*s)")
plt.ylabel("ampiezze")
plt.title(f"{name}")
plt.grid(which="both", linestyle="--")
plt.show()


#funzione di fit multigaussiana
def MGF(x, *params):
    result = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        mu = params[i +1]
        sigma = params[i+2]
        result += a * np.exp(-((x-mu)**2)/(2*(sigma**2)))
    return result

guess = [161, 0.39, 2,
         268, 2.51, 3,
         260, 4.55, 3,
         270, 6.63, 3,
         185, 8.81, 3,
         125.8, 10.83, 3,
         74, 12.9, 5]
          
popt, pcov = curve_fit(MGF, x_data, y_data, p0= guess)

peaks = []
peaks_err = []
amplitudes = []
amplitudes_err = []
sigmas = []
sigmas_err = []
for i in range(7):
    peaks.append(popt[1 + i*3])
    peaks_err.append(np.sqrt(pcov[1 +i*3,1 +i*3]))
    sigmas.append(popt[2+i*3])
    sigmas_err.append(np.sqrt(pcov[2 + i*3, 2 + i*3]))
    amplitudes.append(popt[0+i*3])
    amplitudes_err.append(np.sqrt(pcov[0 + i*3, 0 + i*3]))


picchi= [f"{x}+/-{y}" for x,y in zip(peaks, peaks_err)]
print("---------")
print(f"Posizioni dei picchi{picchi}")
print("---------")

plt.plot(x_data, MGF(x_data,*popt), color="purple")
plt.scatter(x_data, y_data, color="blue", marker=".")
plt.xlabel("area (nV*s)")
plt.ylabel("ampiezze")
plt.title(f"{name}")
plt.grid(which="both", linestyle="--")
plt.show()

#integrale dell'area sotto al picco
def sigma_err_rel(amp):
    return np.sqrt(2*np.pi)*amp

def amp_err_rel(sigma):
    return np.sqrt(2*np.pi)*sigma

areas = []
areas_err = []
for i in range(0,len(popt),3):
    amp = popt[i]
    mu = popt[i+1]
    sigma = popt[i+2]
    area = amp*np.sqrt(2*np.pi)*sigma
    err = area*np.sqrt((np.sqrt(pcov[i,i])/amp)**2 +(np.sqrt(pcov[i+2,i+2])/sigma)**2)
    areas.append(area)
    areas_err.append(err)

g_amp =5000 #V/A
q = 1.6e-19 #carica dell'elettrone
n = 1e-9 #era in nano Vs
c = n/q

charge = []
charge_err = []
for i in range (len(peaks)-2):
    a = (peaks[i+1] - peaks[0])*c/(g_amp)
    b = (peaks_err[i+1]+peaks_err[0])*c/g_amp
    charge.append(a)
    charge_err.append(b)

print(f"AIUTOOOOO\n{charge}{charge_err}")

index = np.linspace(1, len(charge), len(charge))
#plot della retta di guadagno
def linear(x, a , b):
    return x*a +b
popt, pcov = curve_fit(linear, index, charge)

plt.errorbar(index, charge, yerr= charge_err, color="green",fmt="o")
plt.plot(index, linear(index,*popt), color="blue")
plt.legend(title= f"gain ={np.round(popt[0],2)}+-{np.round(np.sqrt(pcov[0,0]),2)}")
plt.grid(which="both", linestyle="--")
plt.title(f"{name}")
plt.show()





