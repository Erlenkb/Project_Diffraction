import numpy as np
import matplotlib.pyplot as plt
import os

####### Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################

######## Third octave settings ########
x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000]
x_ticks_third_octave_labels = ["100", "200", "500", "1k", "2k", "5k"]

third_octave_center_frequencies = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000]

third_octave_lower = [11.2, 14.1, 17.8, 22.4, 28.2, 35.5, 44.7, 56.2, 70.8, 89.1, 112, 141, 178, 224, 282, 355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467, 5623, 7079, 8913, 11220, 14130, 17780,22390]

######################################

###### IR settings ###################
x_ticks_IR = np.linspace(0.0085,0.01,4)
x_ticks_IR_labels = [str(round(i*1000,5)) for i in x_ticks_IR]


length_time = 0.1
start_time = 0.0085
fs = 48000

start = int(fs * start_time)
stop = int(fs * length_time)

path_exposed = "C:/Users/erlen/Project_Diffraction/Code/Data/EXPOSED INSULATION"
path_plywood = "C:/Users/erlen/Project_Diffraction/Code/Data/PLYWOD FRONT"


#print(os.listdir(path_exposed))

def _nextpow2(i):
    n = 1
    while n < i : n*=2
    return n

""" 
count = 0
folder = "C:/Users/erlen/Project_Diffraction/Code/Data/PLYWOD FRONT/Freq Meas/"
for file_name in os.listdir(folder):
    # Construct old file name
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + str(count) + "_S01_R01.etx"

    # Renaming the file
    os.rename(source, destination)
    count += 1
"""

def _getFFT(arr):
    sp = np.pad(arr, (0,_nextpow2(len(arr))-len(arr)),"constant")
    sp = np.fft.fft(sp, _nextpow2(len(arr)))
    sp = np.trim_zeros(sp, trim="fb")
    freq = np.fft.fftfreq(n=len(sp), d=1/fs)
    return np.fft.fftshift(freq), np.fft.fftshift(sp)

for i in range(36):
    file_exposed = "{0}/IR Meas/{1}_S01_R01.etx".format(path_exposed,i)
    file_covered = "{0}/IR Meas/{1}_S01_R01.etx".format(path_plywood,i)
    IR_Covered = np.loadtxt(file_covered, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
    IR_Exposed = np.loadtxt(file_exposed, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
    
    freq_Exposed, fft_exposed = _getFFT(IR_Exposed[start:stop,1])
    freq_Covered, fft_Covered = _getFFT(IR_Covered[start:stop,1])

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    ax.plot(IR_Exposed[:,0],IR_Exposed[:,1], label="Exposed insulation")
    ax.plot(IR_Covered[:,0],IR_Covered[:,1], label="Covered insulation")
    ax.set_title("Degrees of rotation: {0}".format(5*i))
    ax.set_xlim(0.0085,0.01)
    ax.set_xticks(x_ticks_IR)
    ax.set_xticklabels(x_ticks_IR_labels)
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time [ms]")
    ax.grid()
    ax.legend()

    ax1.semilogx(freq_Exposed, np.abs(fft_exposed), label="Exposed insulation")
    ax1.semilogx(freq_Covered, np.abs(fft_Covered), label="Covered insulation")
    ax1.set_title("Degrees of rotation: {0}".format(5*i))
    ax1.set_xlim(100,4000)
    ax1.set_xscale("log")
    ax1.set_xticks(x_ticks_third_octave)
    ax1.set_xticklabels(x_ticks_third_octave_labels)
    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle=":")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude")
    ax1.legend()

    #fig.savefig("Pictures/Angle_Plot_Covered_Vs_Exposed_{0}_Deg.png".format(i*5))
    #plt.close(fig)
    plt.show()
#print(Exposed_IR)


