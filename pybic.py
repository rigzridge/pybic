import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class BicAn:
# Bicoherence analysis class for DSP
    
    FontSize  = 20
    WarnSize  = 1000
    MaxRes    = []
    Samples   = []
    NFreq     = []
    RunBicAn  = False
    NormToNyq = False
    Nseries   = []
    WinVec    = []

    Raw       = []
    Processed = []
    History   = ' '
    SampRate  = 1
    FreqRes   = 0
    SubInt    = 512
    Step      = 128
    Window    = 'hann'       
    Sigma     = 1
    JustSpec  = False
    SpecType  = 'stft'
    ErrLim    = 1e15
    FScale    = 0
    TScale    = 0
    Filter    = 'none'
    Bispectro = False
    Smooth    = 1
    PlotIt    = False
    LilGuy    = 1e-6
    SizeWarn  = True
    CMap      = 'viridis'
    CbarNorth = True
    PlotType  = 'bicoh'
    ScaleAxes = 'manual'
    Verbose   = True
    Detrend   = False
    ZPad      = False
    Cross     = False
    Vector    = False
    TZero     = 0
    tv = [] # Time vector
    fv = [] # Frequency vector
    ff = [] # Full frequency vector
    ft = [] # Fourier amplitudes
    sg = [] # Spectrogram (complex)
    xs = [] # Cross-spectrum
    xc = [] # Cross-coherence
    cs = [] # Coherence spectrum
    bs = [] # Bispectrum
    bc = [] # Bicoherence spectrum
    bp = [] # Biphase proxy
    bg = [] # Bispectrogram
    er = [] # Mean & std dev of FFT
    mb = [] # Mean b^2
    sb = [] # Std dev of b^2

    # Methods
    def __init__(bic,raw):
    # ------------------
    # Constructor
    # ------------------
        bic.Raw = raw


def PlotTest(t,sig):
# ------------------
# Debugger for plots
# ------------------
    fig = plt.figure()

    plt.plot(t,sig)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [arb.]")
    plt.title("Test signal!")

    cid = fig.canvas.mpl_connect('button_press_event', GetClick)

    plt.show()


def ImageTest(dats):
# ------------------
# Debugger for plots
# ------------------
    fig, ax = plt.subplots()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x,y)

    im = ax.imshow(dats, cmap='plasma')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    #plt.xlabel("Time [s]")
    #plt.ylabel("Amplitude [arb.]")
    #plt.title("Test signal!")

    #fig.set_size_inches(18, 16)

    cid = fig.canvas.mpl_connect('button_press_event', GetClick)

    ax.invert_yaxis()
    plt.show()
        
        
def SignalGen(fS,tend,Ax,fx,Afx,Ay,fy,Afy,Az,Ff,noisy):
# ------------------
# Provides FM test signal
# ------------------        
    t = np.arange(0,tend,1/fS)  # Time-vector sampled at "fS" Hz

    # Make 3 sinusoidal signals...
    dfx = Afx*np.sin(2*np.pi*t*Ff)  
    dfy = Afy*np.cos(2*np.pi*t*Ff)
    x = Ax*np.sin( 2*np.pi*(fx*t + dfx) )              # f1
    y = Ay*np.sin( 2*np.pi*(fy*t + dfy) )              # f2
    z = Az*np.sin( 2*np.pi*(fx*t + fy*t + dfx + dfy) ) # f1 + f2

    sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)
    return sig, t, fS


def ApplySTFT(sig,samprate,subint,step,nfreq,t0,detrend,errlim):
# ------------------
# STFT static method
# ------------------
    N = 1
    M = 1 + (len(sig) - subint)//step 
    lim  = nfreq//2                 # lim = |_ Nyquist/res _|
    time_vec = np.zeros(M)          # Time vector
    err  = np.zeros((N,M))          # Mean information
    spec = np.zeros((lim,M,N))      # Spectrogram
    fft_coeffs = np.zeros((N,lim))  # Coeffs for slice
    afft = np.zeros((N,lim))        # Coeffs for slice
    Ntoss = 0                       # Number of removed slices
    
    win = np.sin(np.pi*np.arange(nfreq)/(nfreq-1)) # Apply Hann window
    
    print(' Working...      ')
    for m in range(M):
        LoadBar(m,M-1)

        time_vec[m] = t0 + m*step/samprate
        for k in range(N):
            Ym = sig[m*step : m*step + subint] # Select subinterval 
            #Ym = sig[k,0:subint-1 + m*step] # Select subinterval     
            Ym = Ym[0:nfreq]            # Take only what is needed for res
            if detrend:                 # Remove linear least-squares fit
                dumx = np.arange(1,nfreq+1) 
                dumxy = dumx*Ym
                s = (6/(nfreq*(nfreq**2-1)))*(2*dumxy.sum() - Ym.sum()*(nfreq+1))
                Ym = Ym - s*dumx
            mean = Ym.sum()/len(Ym)
            Ym = win*(Ym-mean)  # Remove DC offset, multiply by window

            DFT = np.fft.fft(Ym)/nfreq  # Limit and normalize by vector length

            fft_coeffs[k,0:lim] = DFT[0:lim]  # Get interested parties
            dumft    = np.abs(fft_coeffs[k,:])    # Dummy for abs(coeffs)
            err[k,m] = dumft.sum()/len(dumft)     # Mean of PSD slice

            if err[k,m]>=errlim:
                fft_coeffs[k,:] = 0*fft_coeffs[k,:] # Blank if mean excessive
                Ntoss += 1

            afft[k,:]  += dumft                   # Welch's PSD
            spec[:,m,k] = fft_coeffs[k,:]         # Build spectrogram

    print('\b\b\b^]\n')
    
    freq_vec = np.arange(nfreq)*samprate/nfreq
    freq_vec = freq_vec[0:lim] 
    afft /= M     
    return spec,afft,freq_vec,time_vec,err,Ntoss


def ApplyCWT(sig,samprate,sigma):
# ------------------
# Wavelet static method
# ------------------
    Nsig = len(sig)
    nyq  = Nsig//2

    f0 = samprate/Nsig
    freq_vec = np.arange(nyq)*f0
    
    CWT = np.zeros((nyq,nyq))

    fft_sig = np.fft.fft(sig)
    fft_sig = fft_sig[0:nyq]

    # Morlet wavelet in frequency space
    Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma/a) * np.exp( -2 * np.pi**2 * sigma**2 * ( freq_vec/a - f0)**2 )

    print(' Working...      ')
    for a in range(nyq):
        LoadBar(a,nyq-1)
        # Apply for each scale (read: frequency)
        CWT[a,:] = np.fft.ifft(fft_sig * Psi(a+1))
    print('\b\b\b^]\n')

    time_vec = np.arange(0,Nsig,2)/samprate
    return CWT,freq_vec,time_vec


def LoadBar(m,M):
# ------------------
# Help the user out!
# ------------------
    ch1 = '||\-/|||'
    ch2 = '_.:"^":.'
    buf = '\b\b\b\b\b\b\b%3.0f%%%s%s' % (100*m/M, ch1[m%8], ch2[m%8])
    print(buf)


def GetClick(event):
# ------------------
# Callback for clicks
# ------------------
    try:
        global clickx, clicky
        clickx, clicky = event.xdata, event.ydata
        buf = 'x = %3.3f, y = %3.3f' % (clickx, clicky)
        print(buf)
    except TypeError:
        print('You can''t click outside the axis!')
    return
