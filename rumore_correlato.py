#rumore correlato 
import lecroyparser
from matplotlib import pyplot as plt
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\SiPMeScintillatoriPlastici\Morrocchi_SiPM\55V\C2--RumoreCorrelato55V--00000.trc"
waveform = lecroyparser.ScopeData(file_path)

#estrai tempo e ampiezza
time = waveform.x
amplitude = waveform.y

plt.plot(time,amplitude)
plt.grid()
plt.show( )

