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

third_octave_center_frequencies = [100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000] #, 6300, 8000, 10000, 12500, 16000, 20000]

third_octave_lower = [89.1, 112, 141, 178, 224, 282, 355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467, 5623]#, 7079, 8913, 11220, 14130, 17780,22390]

######################################

###### IR settings ###################
x_ticks_IR = np.linspace(0.005,0.01,5)
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

def _Lp_from_third_oct(arr):
    """Return the sound pressure level from the third octave band array
    Args:
        arr (_type_): Third octave band array
    Returns:
        _type_: Lp
    """
    Lp = 0
    for i in arr: Lp += i 
    return 10*np.log10(Lp)

def _fft_signal(array):
    N = _nextpow2(len(array))
    array = np.pad(array, (0,_nextpow2(len(array))-len(array)),"constant")
    y = np.fft.fft(array, N)[0:int(N/2)]/N
    #y = _runningMeanFast(y,2)
    
    f = N*np.arange((N/2))/N
    return f, y


def _getFFT(arr):
    sp = np.pad(arr, (0,_nextpow2(len(arr))-len(arr)),"constant")
    sp = np.fft.fft(sp, _nextpow2(len(arr)))
    sp = np.trim_zeros(sp, trim="fb")
    freq = np.fft.fftfreq(n=len(sp), d=1/fs)
    return np.fft.fftshift(freq), np.fft.fftshift(sp)

def _third_Octave_bands(freq, arr, third_oct):
    third_octave_banks = []
    single_bank = []
    i = 0
    for it, x in enumerate(freq):
        if (x > third_oct[-1]) : break
        if (x >= third_oct[i] and x < third_oct[i+1]): 
            single_bank.append(round(arr[it],2))
        if (x >= third_oct[i+1]):
            third_octave_banks.append(single_bank)
            i += 1
            single_bank = []
            single_bank.append(round(arr[it],2))
    filtered_array = []
    for n in third_octave_banks : filtered_array.append(_Lp_from_third_oct(n))
    return filtered_array


def _new_data():
    for i in range(6):
        start1 = int(fs * 0.0083)
        stop1 = int(fs * 0.01)
        path1 = "C:/Users/erlen/Project_Diffraction/Code/Data/New_Data"
        path2 = "C:/Users/erlen/Project_Diffraction/Code/Data/New_Data"
        file1 = "{0}/M002{1}_S01_R01.etx".format(path1,i)
        file2 = "{0}/M003{1}_S01_R01.etx".format(path2,i)

        IR1 = np.loadtxt(file1, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR2 = np.loadtxt(file2, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")




        freq1, fft1 = _getFFT(np.abs(IR1[start1:stop1,1]))
        freq2, fft2 = _getFFT(np.abs(IR2[start1:stop1,1]))
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        ax.plot(IR1[:,0],IR1[:,1], label="Covered")
        ax.plot(IR2[:,0],IR2[:,1], label="Exposed")
        ax.set_title("Degrees of rotation: {0}".format(5*i))
        ax.set_xlim(start1,0.01)
        #ax.set_xticks(x_ticks_IR)
        #ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()

        ax1.semilogx(freq1, np.abs(fft1), label="Exposed insulation")
        ax1.semilogx(freq2, np.abs(fft2), label="Covered insulation")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_xlim(100,4000)
        #ax1.set_xscale("log")
        #ax1.set_xticks(x_ticks_third_octave)
        #ax1.set_xticklabels(x_ticks_third_octave_labels)
        #ax1.grid(which="major")
        #ax1.grid(which="minor", linestyle=":")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude")
        ax1.legend()

        plt.show()



#_new_data()



for i in range(36):
    file_exposed = "{0}/IR Meas/{1}_S01_R01.etx".format(path_exposed,i)
    file_covered = "{0}/IR Meas/{1}_S01_R01.etx".format(path_plywood,i)
    IR_Covered = np.loadtxt(file_covered, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
    IR_Exposed = np.loadtxt(file_exposed, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
    
    freq_Exposed, fft_exposed = _getFFT(np.abs(IR_Exposed[start:stop,1]))
    freq_Covered, fft_Covered = _getFFT(np.abs(IR_Covered[start:stop,1]))

    third_oct_exposed = _third_Octave_bands(freq_Exposed, np.abs(fft_exposed), third_octave_lower)
    third_oct_covered = _third_Octave_bands(freq_Covered, np.abs(fft_Covered), third_octave_lower)


    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    ax.plot(IR_Exposed[:,0],IR_Exposed[:,1], label="Exposed insulation")
    ax.plot(IR_Covered[:,0],IR_Covered[:,1], label="Covered insulation")
    ax.set_title("Degrees of rotation: {0}".format(5*i))
    ax.set_xlim(0.0084,0.015)
    #ax.set_xticks(x_ticks_IR)
    #ax.set_xticklabels(x_ticks_IR_labels)
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time [ms]")
    ax.grid()
    ax.legend()

    ax1.semilogx(freq_Exposed, np.abs(fft_exposed), label="Exposed insulation")
    ax1.semilogx(freq_Covered, np.abs(fft_Covered), label="Covered insulation")
    #ax1.semilogx(freq_Covered, (np.abs(fft_Covered)-np.abs(fft_exposed)), label="Difference Delta dB")
    ax1.set_title("Degrees of rotation: {0}".format(5*i))
    ax1.set_xlim(100,4000)
    #ax1.set_xscale("log")
    #ax1.set_xticks(x_ticks_third_octave)
    #ax1.set_xticklabels(x_ticks_third_octave_labels)
    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle=":")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude")
    ax1.legend()

    #fig.savefig("Pictures/Angle_Plot_Covered_Vs_Exposed_{0}_Deg.png".format(i*5))
    #plt.close(fig)
    plt.show()
#print(Exposed_IR)


