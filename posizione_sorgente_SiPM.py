#posizione_sorgente_SiPM

#quello che voglio fare è riconoscere gli eventi di coincidenza
#per individuare la posizione ho bisogno di conoscere la velocità di propagazione della luce
#nella sbarra e la differenza dei tempi fra i canali 1 e 2

#posizione_sorgenteNa22
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.colors as mcolors

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_1.txt"
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_2.txt"
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\posizioni sorgente 22Na\Position_3.txt"

T_ch1, T_ch2,T_ch3, T_ch4, A_ch1, A_ch2, A_ch3, A_ch4 = np.loadtxt(file_path, unpack = True)

#suppongo che abbiamo acquisito ogni coincidenza
#sottraggo i tempi per trovarmi i ritardi 
delay_1 = T_ch1 - T_ch2
delay_2 = T_ch3 - T_ch4

energia1 = A_ch1 + A_ch2
energia2 =  A_ch3 + A_ch4

bins1 = np.arange(min(energia1), max(energia1) + 1, 1)
bins2 = np.arange(min(energia2), max(energia2) + 1, 1)
plt.hist(energia1, bins=100,histtype='step', color="blue")
plt.hist(energia2, bins=100 , histtype='step', color="red")
plt.show()

trash_hold = (energia1 <100) & (energia1>0)
energia1_filtrata = energia1[trash_hold]
energia2_filtrata = energia2[trash_hold]
delay_1 = delay_1[trash_hold]
delay_2 = delay_2[trash_hold]
print(len(delay_1))

# Calcolo della posizione di interazione sulle due lastre
c = 7 #cm/ns
x_lastra1 = (c * delay_1) /2
x_lastra2 = (c * delay_2) /2

mask_1 = (x_lastra1<22) & (x_lastra1>-22)
mask_2 = (x_lastra2<22) & (x_lastra2>-22)

x_lastra1 = x_lastra1[mask_1]
x_lastra2 = x_lastra2[mask_2]
delay_1 = delay_1[mask_1]
delay_2 = delay_2[mask_2]

b1 =  np.arange(min(x_lastra1), max(x_lastra1) + 1, 1)
b2 =  np.arange(min(x_lastra2), max(x_lastra2) + 1, 1)
print(np.shape(b1))
plt.hist(x_lastra1, bins=b1, histtype='step', color="blue")
plt.hist(x_lastra2, bins=b2, histtype='step', color="red")
plt.xlim(-22,22)
plt.show()

# Definizione delle coordinate delle lastre
y_lastra2 = np.zeros_like(x_lastra2)  # Prima lastra a y = 0 cm
y_lastra1 = np.full_like(x_lastra1, 22)  # Seconda lastra a y = 22 cm

"""
# Disegno delle linee di connessione tra i punti di interazione
plt.plot([-22, 22], [0, 0], 'k-', linewidth=2, label="Lastra 1")
plt.plot([-22, 22], [22, 22], 'k-', linewidth=2, label="Lastra 2")

for i in range(len(x_lastra1)):
    plt.plot([x_lastra1[i], x_lastra2[i]], [y_lastra1[i], y_lastra2[i]], 'g--')
plt.show()
"""
# Metodo Monte Carlo per migliorare la stima della posizione della sorgente
N_sim = 1000  # Numero di simulazioni Monte Carlo
sigma_tempo = 0.2  # Incertezza nei tempi di arrivo (ns)

x_intersezioni = []
y_intersezioni = []

for _ in range(N_sim):
    # Aggiungo rumore ai ritardi temporali
    delay_1_noisy = delay_1 + np.random.normal(0, sigma_tempo, size=len(delay_1))
    delay_2_noisy = delay_2 + np.random.normal(0, sigma_tempo, size=len(delay_2))
    
    # Ricalcolo le posizioni di interazione
    x_lastra1_noisy = (c * delay_1_noisy) / 2
    x_lastra2_noisy = (c * delay_2_noisy) / 2
    
    for i in range(len(x_lastra1_noisy)):
        x1, y1 = x_lastra1_noisy[i], y_lastra1[i]
        x2, y2 = x_lastra2_noisy[i], y_lastra2[i]
        
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            q = y1 - m * x1
            x_intersezione = -q / m
            y_intersezione = m * x_intersezione + q
            
            # Vincoliamo l'intersezione all'interno delle barre
            if -22 <= x_intersezione <= 22 and 0 <= y_intersezione <= 22:
                x_intersezioni.append(x_intersezione)
                y_intersezioni.append(y_intersezione)

# Calcoliamo la posizione media e la deviazione standard della sorgente
x_sorgente = np.mean(x_intersezioni)
y_sorgente = np.mean(y_intersezioni)
sigma_x = np.std(x_intersezioni)
sigma_y = np.std(y_intersezioni)

# Visualizzazione
plt.figure(figsize=(6, 6))
plt.plot([-22, 22], [0, 0], 'k-', linewidth=2, label="Lastra 1")
plt.plot([-22, 22], [22, 22], 'k-', linewidth=2, label="Lastra 2")

plt.scatter(x_intersezioni, y_intersezioni, color='gray', alpha=0.1, s=10, label='Simulazioni Monte Carlo')
plt.scatter(x_sorgente, y_sorgente, color='red', marker='x', s=100, label='Posizione stimata sorgente')
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.title("Ricostruzione della posizione della sorgente (Monte Carlo)")
plt.grid()
plt.show()

print(f"Posizione stimata della sorgente: X = {x_sorgente:.2f} ± {sigma_x:.2f} cm, Y = {y_sorgente:.2f} ± {sigma_y:.2f} cm")