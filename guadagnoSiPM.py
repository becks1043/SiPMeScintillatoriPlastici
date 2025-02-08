import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import csv

path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\F1--Guadagno55V--00000.txt"
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
#guess per 59V ok
"""
guess = [66.6, 0.03, 0.2,
         151.3, 3.49,0.2,
         199.0, 7.22, 0.2,
         156, 10.72, 0.1,
         102.7, 14.84, 0.1,
         81.8, 18.36, 0.5]
"""
#guess per 58V ok
"""
guess = [192.1, 1.70, 0.25,
         213.8, 4.94, 0.25,
         182.6, 8.17, 0.25,
         119.7, 11.48, 0.25,
         68.1, 14.60, 0.25,
         35.0, 18.95, 0.5]
"""
#guess per 57V PERFETTI
"""
guess = [255.9, 1.08, 0.5,
         258.3, 3.70, 0.5,
         167.4, 6.39, 0.5,
         91.4, 8.99, 0.5,
         50.7, 11.6, 0.5,
         21.4, 14.25, 0.5]
"""
#guess per 56V NON TOCCARE
"""
guess = [161, 0.39, 0.5,
         268, 2.51, 0.5,
         260, 4.55, 0.5,
         270, 6.63, 0.5,
         185, 8.81, 0.5,
         125.8, 10.83, 0.5,
         74, 12.9, 0.5]
"""
#guess per 55V top

guess= [80.5, 0.48, 0.5,
        115.1, 2.1, 0.5,
        140.1, 3.6, 0.5,
        114.2, 5.1, 0.5,
        106, 6.6, 0.5,
        60.5, 8.19, 0.5,
        37.7, 9.66, 0.5,
        20.9, 11.14, 0.5,
        15.1, 12.61, 0.5]

#guess per 54V ok
"""
guess = [105.1, 0.30, 0.45,
         105.6, 1.15, 0.45,
         56.4, 2.18, 0.45,
         31, 2.85, 0.4,
         16.7, 3.7, 0.4,
         6.8, 4.55, 0.4 ]
"""
#guess per 53_5V NON é UTILIZZABILE

n_peak =  9 #int
bound = (0, np.inf)
popt, pcov = curve_fit(MGF, x_data, y_data, p0= guess, maxfev = 100000, bounds=bound)

peaks = []
peaks_err = []
amplitudes = []
amplitudes_err = []
sigmas = []
sigmas_err = []
for i in range(n_peak):
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

print(f"Carica generata dalla singola microcella\n{charge}{charge_err}")

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





