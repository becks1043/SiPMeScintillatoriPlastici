# Posizione Sorgente Na22

#quello che voglio fare è riconoscere gli eventi di coincidenza
#per individuare la posizione ho bisogno di conoscere la velocità di propagazione della luce
#nella sbarra e la differenza dei tempi fra i canali 1 e 2

import os 
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, a , mu, sigma):
    return a*np.exp(-(mu-x)**2/(2*sigma**2))

def par(x, a, b, c):
     return a*x**2 + b*x + c

def linear(x, a, b):
     return a*x + b

def main():
    file_path = r"C:\\Users\\user\\Desktop\\Programming\\es4_SiPM\\sorgente-22Na\\Position_"

    xfin, dxfin, yfin, dyfin, x2fin, dx2fin, y2fin, dy2fin = [], [], [], [], [], [], [], []
    
    for k in range(3):
        print(f"\nAnalisi per la posizione incognita n.{k+1}\n\n")
        T_ch1, T_ch2,T_ch3, T_ch4, A_ch1, A_ch2, A_ch3, A_ch4 = np.loadtxt(file_path+str(k+1)+'.txt', unpack = True)
        
        #suppongo che abbiamo acquisito ogni coincidenza
        #sottraggo i tempi per trovarmi i ritardi 
        delay_1 = T_ch2 - T_ch1
        delay_2 = T_ch4 - T_ch3
        energia1 = A_ch1 + A_ch2
        energia2 =  A_ch3 + A_ch4
    
        # Istogramma grezzo per visualizzare le energie lette
        bins1 = np.arange(min(energia1), max(energia1) + 1, 1)
        bins2 = np.arange(min(energia2), max(energia2) + 1, 1)
        plt.hist(energia1, bins=100, histtype='step', color="blue") # barra 'sopra'
        plt.hist(energia2, bins=100, histtype='step', color="red") # barra 'sotto'
        plt.xlim(-100, 1800)
        #plt.legend()
        plt.title("Distribuzione d'energia raccolta nelle barre")
        plt.xlabel("energia cumulata ai capi della barra [u.a.]" , size=15 )
        plt.ylabel("occorrenze" , size=15 )
        plt.savefig("sorgente-22Na\\"
    + 'energia-raccolta' + str(k+1) + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()
        
        # Calcolo della posizione di interazione sulle due lastre, conoscendo la velocità della luce nello 
        # scintillatore
        c = 14.75 #cm/ns
        x_1, x_2 = (c * delay_1)/2, (c * delay_2)/2

        # Per ottenere un risultato fisicamente significativo scartiamo qui tutti i punti per cui
        # le posizioni x risultino esterne ad una delle due barre
        mask = (x_1<22) & (x_1>-22) & (x_2<22) & (x_2>-22)
        x_lastra1, x_lastra2 = x_1[mask], x_2[mask]
        delay_1, delay_2 = delay_1[mask], delay_2[mask]

        #bin_width = 0.0738 
        dt = 0.2 # risoluzione temporale in nanosecondi
        # c è in cm/ns
        ris_spaz = c*dt/2 # da cui la risoluzione spaziale in cm
        bin_width = ris_spaz
        '''
        plt.figure()
        plt.title(f"Na22 in posizione {k+1}, istogramma x1, x2")
        bins1 = np.arange(min(x_lastra1), max(x_lastra1)+ bin_width , bin_width)
        bins2 = np.arange(min(x_lastra2), max(x_lastra2) + bin_width, bin_width)
        plt.hist(x_lastra1, bins=bins1, histtype='step', color="blue", label="barra superiore")
        plt.hist(x_lastra2, bins=bins2, histtype='step', color="red", label="barra inferiore")
        plt.axvline(np.mean(x_lastra1), color='grey', linestyle='--',label="valor medio")
        plt.legend(loc="best", fontsize=15)
        plt.xlabel(r"Posizioni di arrivo dei fotoni $x_1$, $x_2$ [cm]" , size=15 )
        plt.ylabel("occorrenze" , size=15 )
        plt.xlim(-22,22)
        #plt.show()
        plt.close()
        '''
        # fit gaussiano per l'istogramma delle posizioni x sulle barre
        count_x1, edge_x1 = np.histogram(x_lastra1, bins1)
        bin_x1 = (edge_x1[1:] + edge_x1[:-1])*0.5
        guess = [np.max(count_x1), np.mean(count_x1), np.std(count_x1)]
        popt1, pcov1 = curve_fit(gaussian, bin_x1, count_x1, p0=guess, maxfev = 100000)
        x1f = np.linspace(min(bin_x1), max(bin_x1), 10000)
        mu1, sigma1 = popt1[1], popt1[2]
        
        count_x2, edge_x2 = np.histogram(x_lastra2, bins2)
        bin_x2 = (edge_x2[1:] + edge_x2[:-1])*0.5
        guess = [np.max(count_x2), np.mean(count_x2), np.std(count_x2)]
        popt2, pcov2 = curve_fit(gaussian, bin_x2, count_x2, p0=guess, maxfev = 100000)
        x2f = np.linspace(min(bin_x2), max(bin_x2), 10000)
        mu2, sigma2 = popt2[1], popt2[2]
        
        #plot del fit
        plt.plot(x1f, gaussian(x1f, *popt1), color="black", zorder= 5)
        plt.plot(x2f, gaussian(x2f, *popt2), color="black", zorder= 5)
        plt.hist(x_lastra1, bins=bins1, histtype='step', color="blue") # barra sopra
        plt.hist(x_lastra2, bins=bins2, histtype='step', color="red") # barra sotto
        plt.axvline(mu1, color='black', linestyle='--',label=f"$\mu$={np.round(mu1, 2)}$\pm${np.round(sigma1)} cm")
        plt.axvline(mu2, color='black', linestyle='--',label=f"$\mu$={np.round(mu2, 2)}$\pm${np.round(sigma2)} cm")
        plt.legend(loc="best", fontsize=15)
        plt.xlim(-22,22)
        plt.xlabel(r"Posizioni di arrivo dei fotoni $x_1$, $x_2$ [cm]" , size=15 )
        plt.ylabel("occorrenze" , size=15 )
        plt.savefig("sorgente-22Na\\"
    + 'posizioni-x-arrivo-fotoni_'+ str(k+1) + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()

        #bestx = (mu1 + mu2) / 2
        #bestdx = np.sqrt( sigma1**2 + sigma2**2 )
        
        # Definizione delle coordinate delle lastre
        y_lastra1 = np.full_like(x_lastra1, +11)  # barra sopra
        y_lastra2 = np.full_like(x_lastra2, -11)  # barra sotto
        
        # Range di y da analizzare
        n = np.shape(x_lastra1)[0]
        yscan = np.arange(-11, 11, bin_width)  # definiamo i bordi e il numero di rette
        # bin_width è uguale alla risoluzione spaziale
        std_x_per_y = []
        std_x_error_per_y = []
        #mean_x_per_y = []
        
        # il ciclo gira per ogni retta ycut parallela alle barre di rivelatore, calcolando 
        # le intersezioni fra ycut e le 
        for ycut in yscan:
            # usiamo x_at_cut per raccogliere le intersezioni fra yscan=np.linspace(11, -11, 50)
            x_at_cut = []
            # ripetiamo il ciclo for per ogni coincidenza i
            for i in range(n):
                if (x_lastra2[i] != x_lastra1[i]):
                    # la retta di volo dei fotoni avrà questa intercetta e coeff angolare a, b:
                    a = (y_lastra2[i] - y_lastra1[i]) / (x_lastra2[i] - x_lastra1[i])
                    b = y_lastra1[i] - a * x_lastra1[i]
                    # pertanto questa è l'intersezione:
                    x = (ycut - b) / a
                    if -22 <= x <= 22:
                        x_at_cut.append(x)
                else:
                    x_at_cut.append(x_lastra2[i])
            if len(x_at_cut) > 1:
                # mean_x = np.mean(x_at_cut)
                std_x = np.std(x_at_cut)
            else:
                std_x = np.nan  # troppo pochi dati
            
            std_x_per_y.append(std_x)
            # mean_x_per_y.append(mean_x)
            # vogliamo anche che ci restituisca l'errore sulla standard deviation
            std_x_error_per_y.append( std_x/np.sqrt(2*(len(x_at_cut) - 1)) )

        # convertiamo i risultati del ciclo for in array
        std_x_per_y = np.array(std_x_per_y)
        std_errors = np.array(std_x_error_per_y)
        
        # trovo l'indice in cui std_x_per_y è minimo con np.nanargmin(std_x_per_y)
        # ho così la migliore y, best_y, attorno a cui lavorare (vedi fit minimiz. parabolico)
        best_y = yscan[np.nanargmin(std_x_per_y)]

        
        # FIT minimiz. parabolico
        window = 5.0  # o 3.0, puoi provarli entrambi
        mask = (yscan >= best_y - window) & (yscan <= best_y + window)
        yscan_fit = yscan[mask]
        std_fit = std_x_per_y[mask]
        errors_fit = std_errors[mask]
        
        guess2 = [1, -0.2, np.min(std_fit)]
        poptpar, pcovpar = curve_fit(par, yscan_fit, std_fit, sigma=errors_fit, p0= guess2)
        x2 = np.linspace(min(yscan), max(yscan), 1000)
        
        a, b = poptpar[0], poptpar[1]
        min_y = -b/(2*a)
        Sa, Sb = np.sqrt(pcovpar[0, 0]), np.sqrt(pcovpar[1, 1])
        min_y_err = np.sqrt((Sb / (2*a))**2 + (b * Sa / (2*a**2))**2)
        # (arriva dal fatto che per una funzione f(a, b) la propagazione dell'errore sarà
        # sigma_f = sqrt{ [(sigmaA)^2 * (df/da)^2] + [(sigmaB)^2 * (df/db)^2] } )

        # Troviamo Mean/Std di Xi nella miglior y trovata, così da avere bestx e bestdx
        ycut = min_y
        x_at_cut = []
        for i in range(n):
                if (x_lastra2[i] != x_lastra1[i]):
                    # la retta di volo dei fotoni avrà questa intercetta e coeff angolare a, b:
                    a = (y_lastra2[i] - y_lastra1[i]) / (x_lastra2[i] - x_lastra1[i])
                    b = y_lastra1[i] - a * x_lastra1[i]
                    # pertanto questa è l'intersezione:
                    x = (ycut - b) / a
                    if -22 <= x <= 22:
                        x_at_cut.append(x)
        if len(x_at_cut) > 1:
                bestx = np.mean(x_at_cut)
                bestdx = np.std(x_at_cut)
        else:
                std_x = np.nan  # troppo pochi dati

        print("primo metodo, x, y, dx, dy:")
        print('x: ',bestx,'\ny: ', best_y)
        print('dx: ',bestdx,'\ndy: ', min_y_err)
        
        # PLOT della minimizzazione della deviazione standard delle intersezioni
        plt.figure(figsize=(8, 5))
        plt.plot(x2, par(x2, *poptpar), color="darkgreen")
        plt.errorbar(yscan, std_x_per_y, std_errors, fmt='+', label='Deviazione standard di x sulle rette y')
        plt.axvline(min_y, color='red', linestyle='--', label=f'y_min = {min_y:.3f}$\pm${min_y_err:.3f} cm')
        plt.xlabel('y [cm]', size=15)
        plt.ylabel(r'$\sigma$ delle intersezioni x', size=15)
        plt.title(f'Posizione y della sorgente nella posizione {k+1}')
        plt.legend(fontsize=15)
        plt.grid(linestyle="--")
        plt.savefig("sorgente-22Na\\"
    + 'minimizzazione-sigma_' + str(k+1) + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()

        # Proviamo un secondo metodo, trattiamo tutte queste traiettorie come se fossero un 
        # vero e proprio fascio di rette, di equazione yc= m*xc + q, quindi possiamo trovare
        # le nostre m e q come prima, calcolarne gli errori ed infine fare un fit della
        # retta m = yc/xc - q/xc
        def TrovaCentroFascio(x1, x2, y1, y2):
            
            m, q, dm, dq = [], [], [], []
            
            for i in range(n):
                m.append( (y2[i] - y1[i]) / (x2[i] - x1[i]) )
                q.append( y1[i] - m[i] * x1[i] )
                # ipotizzando sigma di (x_lastra2[i] - x_lastra1[i]) come sqrt(2) derivante
                # dall'ipotesi che x1, x2 abbiano 1cm di errore
                dm.append( abs( 22 / (x2[i] - x1[i])**2 ) * np.sqrt(2) )
                dq.append( np.sqrt( (x1[i]*dm[i])**2 + m[i]**2 ) ) 

            popt, pcov = curve_fit(linear, q, m, sigma=dm, p0=(-1/bestx, -best_y/bestx), absolute_sigma= False)
            
            bestx3, besty3 = -1/popt[0], -popt[1]/popt[0]
            bestdx3, bestdy3 = abs(np.sqrt(np.diag(pcov)[0])/popt[0]**2), np.sqrt( np.diag(pcov)[1]/(popt[0])**2 +np.diag(pcov)[0]*(popt[1]/popt[0]**2)**2 )  

            plt.figure()
            plt.title(f'X={bestx3-100}; Y={besty3-100}')
            plt.scatter(q, m, label="punti (m, q)")
            plt.plot(bestx3, linear(bestx3, *popt), label='fit')
            plt.legend(loc='best')
            #plt.show()
            
            return bestx3-100, besty3-100, bestdx3, bestdy3

        bestx3, besty3, bestdx3, bestdy3 = TrovaCentroFascio(x_lastra1+100, x_lastra2+100, y_lastra1+100, y_lastra2+100)

        print("secondo metodo, x, y, dx, dy:\n")
        print('x: ',bestx3,'\ny: ', besty3)
        print('dx: ',bestdx3,'\ndy: ', bestdy3)
        
        # plot delle posizione k-esima della sorgente far le due barre
        name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(name)
        plt.hlines(y=11, xmin=-22, xmax=22, color="black", linestyle="-")
        plt.hlines(y=-11, xmin=-22, xmax=22, color="black", linestyle="-")
        plt.errorbar(bestx,min_y,min_y_err,bestdx, fmt="+",color = "red", capsize=1, label="pos. sorg.")
        plt.errorbar(bestx3,besty3,bestdy3,bestdx3, fmt="+",color = "red", capsize=2, label="pos. sorg. met. 2")
        plt.grid(linestyle="--")
        plt.legend(fontsize=15)
        plt.title(f"Posizione del Na22 nel caso incognito {k+1}")
        plt.xlabel('posizione parallela alle barre [cm]', size=10)    
        plt.ylabel('posizione perpendicolare alle barre [cm]', size=10)
        plt.savefig("sorgente-22Na\\"
    + 'posizione'+ str(k+1) + ".pdf", format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close()

        xfin.append(bestx)
        dxfin.append(bestdx)
        yfin.append(min_y)
        dyfin.append(min_y_err)

        x2fin.append(bestx3)
        dx2fin.append(bestdx3)
        y2fin.append(besty3)
        dy2fin.append(bestdy3)
        # fine del calcolo sulle 3 posizioni

    #plot delle posizioni della sorgente far le due barre
    name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(name)

    plt.figure(figsize=(9,5))
    plt.hlines(y=11, xmin=-22, xmax=22, color="black", linestyle="-")
    plt.hlines(y=-11, xmin=-22, xmax=22, color="black", linestyle="-")
    for j in range(len(xfin)):
        plt.errorbar(xfin[j], yfin[j], dyfin[j], dxfin[j], fmt="+", capsize=2, label=f"pos. {j+1}, metodo I", zorder=3)
        plt.errorbar(x2fin[j], y2fin[j], dy2fin[j], dx2fin[j], fmt="+", capsize=2, label=f"pos. {j+1}, metodo II", zorder=3)
    plt.xlabel('posizione parallela alle barre [cm]', size=12)
    plt.ylabel('posizione perpendicolare alle barre [cm]', size=12)
    plt.grid(linestyle="--")
    plt.legend(fontsize=12, loc=4)
    #plt.title(r"posizioni incognite del $^{22}$Na stimate")
    plt.savefig("sorgente-22Na\\"
    + 'stima-posizioni' + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()