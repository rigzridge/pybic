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

    # Functions
    def __init__(bic,raw):
    # ------------------
    # Constructor
    # ------------------
        bic.raw = raw

    def PlotTest(t,sig):
    # ------------------
    # Debugger for plots
    # ------------------
        import matplotlib.pyplot as plt

        plt.plot(t,sig)
        plt.xlabel("Time [Hz]")
        plt.ylabel("Amplitude [arb.]")
        plt.title("Test signal!")
        plt.show()


    def SignalGen(fS,tend,Ax,fx,Afx,Ay,fy,Afy,Az,Ff,noisy):
    # ------------------
    # Provides FM test signal
    # ------------------
        import numpy as np
        
        t = np.arange(0,tend,1/fS)  # Time-vector sampled at "fS" Hz

        # Make 3 sinusoidal signals...
        dfx = Afx*np.sin(2*np.pi*t*Ff)  
        dfy = Afy*np.cos(2*np.pi*t*Ff)
        x = Ax*np.sin( 2*np.pi*(fx*t + dfx) )              # f1
        y = Ay*np.sin( 2*np.pi*(fy*t + dfy) )              # f2
        z = Az*np.sin( 2*np.pi*(fx*t + fy*t + dfx + dfy) ) # f1 + f2

        sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)

        return sig, t

        
