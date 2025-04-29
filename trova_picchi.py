import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from lecroyscope import Trace
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from tqdm import tqdm


class DLED:
    def __init__(self, file_path):
        self.file_path = file_path
    def trova_picchi(self):
        trace = Trace(self.file_path)
        time = trace.time  # trace.x is an alias for trace.time
        #time *= 1e3 #tempo in ms
        # channel voltage values
        voltage = trace.voltage  # trace.y is an alias for trace.voltage
        shift = 0.5*1e-9
        f_interp = interp1d(time, voltage, bounds_error=False, fill_value="extrapolate")
        # Calcola il segnale shiftato: aggiungi lo shift ai tempi originali
        v_shifted = f_interp(time + shift)
        # Sottrai il segnale shiftato dall'originale
        diff_signal =  v_shifted - voltage 
        peaks, _ = find_peaks(diff_signal, height= 0.02)  # Trova tutti i picchi
        diff_time = np.diff(time[peaks])
        diff_signal = diff_signal[peaks][1:]
        return diff_time, diff_signal
    
#file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\55V\C2--RumoreCorrelato55V--00000.trc"

# Funzione per elaborare un singolo file
def elabora_file(file_path):
    dled = DLED(file_path)
    diff_time, diff_signal = dled.trova_picchi()
    return diff_time, diff_signal

def elabora_cartella(cartella_path):
    file_list = [f for f in os.listdir(cartella_path) if f.endswith(".trc")]
    diff_time_global, diff_signal_global = [], []

    for filename in tqdm(file_list, desc="Elaborazione file", unit="file"):
            file_path = os.path.join(cartella_path, filename)
            diff_time, diff_signal = elabora_file(file_path)
            diff_time_global.extend(diff_time)
            diff_signal_global.extend(diff_signal)
    return np.array(diff_time_global), np.array(diff_signal_global)

cartella_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\58V"
#cartella_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\56V"
# Esegui l'elaborazione sui file nella cartella
diff_time_global, diff_signal_global = elabora_cartella(cartella_path)
diff_time_global *= 1e9
name = os.path.basename(cartella_path)
#name = os.path.splitext(name)[0]

#maschere 55V
mask1 = (diff_time_global >= 1e1) & (diff_time_global<= 1e2) & (diff_signal_global <= 0.0319) #after pulse
mask2 = (diff_signal_global >= 0.0524) #dark cross talk
mask3 = (diff_signal_global <= 0.0524) & (diff_signal_global >= 0.0319) & (diff_time_global <= 90) #delayed cross talk
mask4 = (diff_signal_global <= 0.0524) & (diff_signal_global >= 0.0319) & (diff_time_global >= 90) #dark count
#maschere a 56V
mask1 = (diff_time_global >= 1) & (diff_time_global<= 1e2) & (diff_signal_global <= 0.047) #after pulse
mask2 = (diff_signal_global >= 0.0641) #dark cross talk
mask3 = (diff_signal_global <= 0.0641) & (diff_signal_global >= 0.047) & (diff_time_global <= 45) #delayed cross talk
mask4 = (diff_signal_global <= 0.0641) & (diff_signal_global >= 0.047) & (diff_time_global >= 90) #dark count
#machere a 57V
mask1 = (diff_time_global >= 5.7) & (diff_time_global<= 1e2) & (diff_signal_global <= 0.0634) #after pulse
mask2 = (diff_signal_global >= 0.08) #dark cross talk
mask3 = (diff_signal_global <= 0.08) & (diff_signal_global >= 0.0634) & (diff_time_global <= 90) #delayed cross talk
mask4 = (diff_signal_global <= 0.08) & (diff_signal_global >= 0.0634) & (diff_time_global >= 90) #dark count
#maschere a 58V (forse per queste tensioni va alzata la soglia di trashold)
mask1 = (diff_time_global >= 15.) & (diff_time_global<= 1e2) & (diff_signal_global <= 0.07)& (diff_signal_global >= 0.0143) #after pulse
mask2 = (diff_signal_global >= 0.10) #dark cross talk
mask3 = (diff_signal_global <= 0.093) & (diff_signal_global >= 0.07) & (diff_time_global <= 90) #delayed cross talk
mask4 = (diff_signal_global <= 0.093) & (diff_signal_global >= 0.07) & (diff_time_global >= 90) #dark count
#maschera a 59V
"""
mask1 = (diff_time_global >= 5.7) & (diff_time_global<= 1e2) & (diff_signal_global <= 0.0634) #after pulse
mask2 = (diff_signal_global >= 0.08) #dark cross talk
mask3 = (diff_signal_global <= 0.08) & (diff_signal_global >= 0.0634) & (diff_time_global <= 90) #delayed cross talk
mask4 = (diff_signal_global <= 0.08) & (diff_signal_global >= 0.0634) & (diff_time_global >= 90) #dark count
"""
#tempo di recovery
t_r = 111*1e-15 * 385*1e3 #tempo di recovery uguale per ogni voltaggio
def linear (x, a ,b):
     return a*x + b

def recharge (t, R, tau):
    return R * (1 - np.exp(-t/tau))

popt1, _ = curve_fit(linear, diff_time_global[mask1], diff_signal_global[mask1])
popt2, _ = curve_fit(linear, diff_time_global[mask4], diff_signal_global[mask4])
popt3, pcov3 = curve_fit(recharge, diff_time_global[mask1] , diff_signal_global[mask1])
#0.0346
print(f"-----")
#print(f"popt3 = {popt3} / {t_r}")
x1 = np.linspace(0, max(diff_time_global[mask1]), 10000)
x2 = np.linspace(0, max(diff_time_global[mask4]), 10000)

print("-----")
print(f"tempo di scatrica microcella {popt3[1]}")
print(f"after pulse ={np.round((sum(diff_time_global[mask1])/sum(diff_time_global))*100, 2)}%")
print(f"cross talk ={np.round((sum(diff_time_global[mask2])/sum(diff_time_global))*100, 2)}%")
print(f"delayed cross talk + AP ={(sum(diff_time_global[mask2])+sum(diff_time_global[mask1]))/sum(diff_time_global)*100:.2f}%")

plt.scatter(diff_time_global , diff_signal_global, s=1, marker="o" ,color ="purple")#, label="dark count"
plt.scatter(diff_time_global[mask1] , diff_signal_global[mask1], s=1, marker="o" ,color ="red")#, label="after pulse"
plt.scatter(diff_time_global[mask2] , diff_signal_global[mask2], s=1, marker="o" ,color ="green")#, label="dark + corss talk"
plt.scatter(diff_time_global[mask3] , diff_signal_global[mask3], s=1, marker="o" ,color ="blue")#, label="delayed cross talk"
#plt.plot(x1, linear(x1, *popt1), color= "orange")
#plt.plot(x2, linear(x2, *popt2), color= "orange")
plt.plot(x2, recharge(x2, *popt3), color= "black")
plt.xscale("log")
plt.xlabel("differenze temporali [ns]")
plt.ylabel("Ampiezza")
plt.grid(linestyle="--")
plt.legend(title= fr"$\tau$ = {popt3[1]:.2f} $\pm$ {np.sqrt(pcov3[1][1]):.2f} ns")
plt.title(f"Rumore correlato con alimentazione a {name}")
plt.show()




