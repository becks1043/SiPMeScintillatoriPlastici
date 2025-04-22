#posizione_sorgente_SiPM

#quello che voglio fare è riconoscere gli eventi di coincidenza
#per individuare la posizione ho bisogno di conoscere la velocità di propagazione della luce
#nella sbarra e la differenza dei tempi fra i canali 1 e 2

#posizione_sorgenteNa22
import os 
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_1.txt"
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_2.txt"
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_3.txt"

T_ch1, T_ch2,T_ch3, T_ch4, A_ch1, A_ch2, A_ch3, A_ch4 = np.loadtxt(file_path, unpack = True)

#suppongo che abbiamo acquisito ogni coincidenza
#sottraggo i tempi per trovarmi i ritardi 
delay_1 =  T_ch2 - T_ch1 
delay_2 =  T_ch4  -  T_ch3 

energia1 = A_ch1 + A_ch2
energia2 =  A_ch3 + A_ch4

bins1 = np.arange(min(energia1), max(energia1) + 1, 1)
bins2 = np.arange(min(energia2), max(energia2) + 1, 1)
plt.hist(energia1, bins=100,histtype='step', color="blue", label="barra sopra")
plt.hist(energia2, bins=100 , histtype='step', color="red", label= "barra sotto")
plt.legend()
plt.title("Distribuzione d'energia nelle barre")
plt.show()

# Calcolo della posizione di interazione sulle due lastre
c = 14.75 #cm/ns
x_lastra1 = (c * delay_1) /2
x_lastra2 = (c * delay_2) /2

mask_1 = (x_lastra1<22) & (x_lastra1>-22) & (x_lastra2<22) & (x_lastra2>-22)
#mask_2 = (x_lastra2<22) & (x_lastra2>-22)

x_lastra1 = x_lastra1[mask_1]
x_lastra2 = x_lastra2[mask_1]
delay_1 = delay_1[mask_1]
delay_2 = delay_2[mask_1]

bin_width = 1.  # ris spaziale del sistema in cm 
bins1 = np.arange(min(x_lastra1), max(x_lastra1) + bin_width , bin_width)
bins2 = np.arange(min(x_lastra2), max(x_lastra2) + bin_width, bin_width)
plt.hist(x_lastra1, bins=bins1, histtype='step', color="blue", label="barra sopra")
plt.hist(x_lastra2, bins=bins2, histtype='step', color="red", label="barra sotto")
plt.axvline(np.mean(x_lastra1), color='blue', linestyle='--',label="valor medio")
plt.legend(loc="best")
plt.xlim(-22,22)
plt.show()
#fit gaussiano per l'istogramma
count_x1, edge_x1 = np.histogram(x_lastra1, bins1)
bin_x1= (edge_x1[1:] + edge_x1[:-1])*0.5

guess = [np.max(count_x1), np.mean(count_x1), np.std(count_x1)]
def gaussian(x, a , mu, sigma):
    return a*np.exp(-(mu-x)**2/(2*sigma**2))
popt, pcov = curve_fit(gaussian, bin_x1, count_x1, p0=guess, maxfev = 100000)
x1 = np.linspace(min(bin_x1), max(bin_x1), 1000)
#plot del fit
plt.plot(x1, gaussian(x1, *popt), color="black", zorder= 5)
plt.hist(x_lastra1, bins=bins1, histtype='step', color="blue", label="barra sopra")
plt.hist(x_lastra2, bins=bins2, histtype='step', color="red", label="barra sotto")
plt.axvline(popt[1], color='black', linestyle='--',label=f"$\mu$={np.round(popt[1], 2)}$\pm${np.round(popt[2])} cm")
plt.legend(loc="best")
plt.xlim(-22,22)
plt.show()

# Definizione delle coordinate delle lastre
y_lastra1 = np.full_like(x_lastra1, +11)  # barra sopra
y_lastra2 = np.full_like(x_lastra2, -11)  # barra sotto

# Range di y da analizzare
n = np.shape(x_lastra1)[0]
yscan = np.linspace(11, -11, 50)  # definiamo i bordi e il numero di rette
std_x_per_y = []
for ycut in yscan:
    x_at_cut = []
    for i in range(n):
        if x_lastra2[i] != x_lastra1[i]:
            a = (y_lastra2[i] - y_lastra1[i]) / (x_lastra2[i] - x_lastra1[i])
            b = y_lastra1[i] - a * x_lastra1[i]
            x = (ycut - b) / a
            if -22 <= x <= 22:
                x_at_cut.append(x)
    if len(x_at_cut) > 1:
        std_x = np.std(x_at_cut)
    else:
        std_x = np.nan  # troppo pochi dati
    std_x_per_y.append(std_x)

std_x_per_y = np.array(std_x_per_y)
best_y = yscan[np.nanargmin(std_x_per_y)]
#fit parabolico
window = 2.0  # o 3.0, puoi provarli entrambi
mask = (yscan >= best_y - window) & (yscan <= best_y + window)
yscan_fit = yscan[mask]
std_fit = std_x_per_y[mask]

def par(x, a, b, c):
     return a*x**2 + b*x + c
guess2 = [1, -0.2, np.min(std_fit)]
popt2, pcov2 = curve_fit(par, yscan_fit, std_fit, p0= guess2)
x2 = np.linspace(min(yscan), max(yscan), 1000)
# Plot della deviazione standard
min_y = -popt2[1]/(2*popt2[0])
min_y_err = min_y*np.sqrt((np.sqrt(pcov2[0,0])/popt2[0])**2+(np.sqrt(pcov2[1,1])/popt2[1])**2)
plt.figure(figsize=(8, 5))
plt.plot(x2, par(x2, *popt2), color="darkgreen")
plt.scatter(yscan, std_x_per_y, label='Deviazione standard di x sulle rette y')
plt.axvline(min_y, color='red', linestyle='--', label=f'Min a y = {min_y:.3f}$\pm${min_y_err:.3f} cm')
plt.xlabel('y [cm]')
plt.ylabel('Deviazione standard delle intersezioni x')
plt.title('Posizione y della sorgente')
plt.legend()
plt.grid(linestyle="--")
plt.show()

#plot delle posizioni della sorgente far le due barre
name = os.path.basename(file_path)
base_name, ext = os.path.splitext(name)
plt.hlines(y=11, xmin=-22, xmax=22, color="black", linestyle="-")
plt.hlines(y=-11, xmin=-22, xmax=22, color="black", linestyle="-")
plt.errorbar(popt[1], min_y, np.sqrt(min_y_err**2+bin_width**2), abs(popt[2]), fmt="+",color = "red", capsize=2, label="posizione sorgente")
plt.grid(linestyle="--")
plt.legend()
plt.title(f"Posizione del Na22 per {base_name}")
plt.show()
