#posizione_sorgenteNa22
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from scipy.optimize import curve_fit

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_2.txt"

T_ch1, T_ch2,T_ch3, T_ch4, A_ch1, A_ch2, A_ch3, A_ch4 = np.loadtxt(file_path, unpack = True)

#suppongo che abbiamo acquisito ogni coincidenza
#sottraggo i tempi per trovarmi i ritardi 
delay_1 = T_ch1 - T_ch2
delay_2 = T_ch3 - T_ch4

energia1 = A_ch1 + A_ch2
energia2 =  A_ch3 + A_ch4

soglia_energia1 = np.percentile(energia1, 10)
soglia_energia2 = np.percentile(energia1, 90)


# Filtriamo gli eventi
mask = (energia1 > soglia_energia1) & (soglia_energia2 < energia1) # True se l'evento supera la soglia
energia1_filtrata = energia1[mask]
energia2_filtrata = energia2[mask]

# Definizione dei parametri
LUNGHEZZA_LASTRA = 44  # cm
DISTANZA_LASTRE = 22  # cm
C_VELOCITA_LUCE = 7  # cm/ns (esempio, cambia con il valore reale)

# Simuliamo un set di dati (sostituiscilo con i tuoi dati reali)
ritardi_lastra1 = T_ch1 - T_ch2 # ns
ritardi_lastra2 = T_ch3 - T_ch4  # ns

ritardi_filtrati1 = ritardi_lastra1[mask]  # Applichiamo lo stesso filtro ai ritardi
ritardi_filtrati2 = ritardi_lastra2[mask]
# Calcolo della posizione di interazione sulle due lastre
x_lastra1 = (C_VELOCITA_LUCE * ritardi_lastra1) / 2
x_lastra2 = (C_VELOCITA_LUCE * ritardi_lastra2) / 2

# Definizione delle coordinate delle lastre
y_lastra1 = np.zeros_like(x_lastra1)  # Prima lastra a y = 0 cm
y_lastra2 = np.full_like(x_lastra2, DISTANZA_LASTRE)  # Seconda lastra a y = 22 cm

# Creazione del grafico
plt.figure(figsize=(6, 8))
plt.xlim(-LUNGHEZZA_LASTRA / 2, LUNGHEZZA_LASTRA / 2)
plt.ylim(-5, DISTANZA_LASTRE + 5)

# Disegno delle lastre
plt.plot([-LUNGHEZZA_LASTRA / 2, LUNGHEZZA_LASTRA / 2], [0, 0], 'k-', linewidth=2, label="Lastra 1")
plt.plot([-LUNGHEZZA_LASTRA / 2, LUNGHEZZA_LASTRA / 2], [DISTANZA_LASTRE, DISTANZA_LASTRE], 'k-', linewidth=2, label="Lastra 2")

# Disegno delle posizioni di interazione
plt.scatter(x_lastra1, y_lastra1, color='r', label="Interazione Lastra 1")
plt.scatter(x_lastra2, y_lastra2, color='b', label="Interazione Lastra 2")

# Disegno delle linee di connessione tra i punti di interazione
for i in range(len(x_lastra1)):
    plt.plot([x_lastra1[i], x_lastra2[i]], [y_lastra1[i], y_lastra2[i]], 'g--')

# Aggiunta delle etichette
plt.xlabel("Posizione sulla lastra (cm)")
plt.ylabel("Distanza tra le lastre (cm)")
plt.title("Tracciamento delle interazioni")
plt.legend()
plt.grid()
plt.show()
