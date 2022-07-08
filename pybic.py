#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       ______     ______ _      
#       | ___ \    | ___ (_)         Bicoherence Analysis Module for Python
#       | |_/ /   _| |_/ /_  ___     --------------------------------------
#       |  __/ | | | ___ \ |/ __|    
#       | |  | |_| | |_/ / | (__               [ v0.9 ] - 2022
#       \_|   \__, \____/|_|\___|
#              __/ |                         G. Riggs | T. Matheny
#             |___/   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                  
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# The Bispectrum
# B_xyz(f1,f2) = < X(f1)Y(f2)Z(f1+f2)* >, where x,y,z are time series with 
# corresponding Fourier transforms X,Y,Z, and <...> denotes averaging.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# The (squared) Bicoherence spectrum
# b^2_xyz(f1,f2) =           |B_xyz(f1,f2)|^2
#                          --------------------
#                ( <|X(f1)Y(f2)|^2> <|Z(f1+f2)|^2> + eps ),
# where eps is a small number meant to prevent 0/0 = NaN catastrophe
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Inputs
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# inData    -> time-series {or structure}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# additional options... (see below for instructions)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# autoscale -> autoscaling in figures [default :: false]
# bispectro -> computes bispectrogram [default :: false]
# cbarnorth -> control bolorbar location [default :: true]
# cmap      -> adjust colormap [default :: viridis]
# dealias   -> applies antialiasing (LP) filter [default :: false]
# detrend   -> remove linear trend from data [default :: false]
# errlim    -> mean(fft) condition [default :: inf] 
# filter    -> xxxxxxxxxxxxxxx [default :: 'none']
# freqres   -> desired frequency resolution [Hz]
# fscale    -> scale for plotting frequencies [default :: 0]
# justspec  -> true for just spectrogram [default :: false]
# lilguy    -> set epsilon [default :: 1e-6]
# note      -> optional string for documentation [default :: {DATE & TIME}] 
# plotit    -> start plotting tool when done [default :: false]
# plottype  -> set desired plottable [default :: 'bicoh']
# samprate  -> sampling rate in Hz [default :: 1]
# sigma     -> parameter for wavelet spectrum [default :: 1]
# spectype  -> set desired time-freq. method [default :: 'stft']
# step      -> step size for Welch method in samples [default :: 512]
# subint    -> subinterval size in samples [default :: 128]
# sizewarn  -> warning for matrix size [default :: true]
# smooth    -> smooths FFT by n samples [default :: 1]
# tscale    -> scale for plotting time [default :: 0]
# verbose   -> allow printing of info structure [default :: true]
# window    -> select window function [default :: 'hann']
# zpad      -> add zero-padding to end of time-series [default :: true]
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Version History
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/06/2022 -> Tyler tackled the tedium of porting static methods over from
# the Matlab version. Bit of debugging, but things are all but error-free.
# Fiddling with plot methods, font sizes, colorbar locations, etc. 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/5/2022 --> Fixed some issues with STFT method; first "tests" attempted.
# Added GetClick method to obtain mouse coordinates on click ~> should be
# incredibly helpful down the road when we're trying to get the GUI up.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/4/2022 --> Copy pasta'd some code from MATLAB class
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/1/2022 --> First "code." 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime

# Define classes for bispec script

class BicAn:
# Bicoherence analysis class for DSP
    
    # Properties
    FontSize  = 20
    WarnSize  = 1000
    Date      = datetime.now()
    MaxRes    = []
    Samples   = []
    NFreq     = []
    RunBicAn  = False
    NormToNyq = False
    Nseries   = 1
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


    def PlotSpectro(self):
    # ------------------
    # Plot spectrograms
    # ------------------
        tstr = 'Time [%ss]' % (ScaleToString(self.TScale))
        fstr = '$f$ [%sHz]' % (ScaleToString(self.FScale))
        if self.SpecType=='stft':
            dum = 'P'
        else:
            dum = 'W'
        cbarstr = '$\log_{10}|\mathcal{%s}(t,f)|^2$' % (dum)

        fig, ax = plt.subplots()
        for k in range(self.Nseries):
            im = ax.pcolor(self.tv/10**self.TScale,self.fv/10**self.FScale,2*np.log10(abs(self.sg[:,:,k])), cmap=self.CMap, shading='auto')
            PlotLabels(fig,[tstr,fstr,cbarstr],self.FontSize,self.CbarNorth,ax,im)

        plt.show()
        return

# Static methods

def PlotLabels(fig,strings,fsize,cbarNorth,ax,im):
# ------------------
# Convenience function
# ------------------
    n = len(strings)
    fweight = 'normal'
    plt.xlabel(strings[0], fontsize=fsize, fontweight=fweight)
    if n>1:
        plt.ylabel(strings[1], fontsize=fsize, fontweight=fweight)
    plt.xticks(size=fsize, weight='bold')
    plt.yticks(size=fsize, weight='bold')
    plt.minorticks_on()
    if n>2:
        divider = make_axes_locatable(ax)
        if cbarNorth:
            cax = divider.append_axes('top', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='horizontal')   
            cax.xaxis.set_label_position('top') 
            cax.xaxis.tick_top()     
            plt.xlabel(strings[2], fontsize=fsize, fontweight=fweight)
            plt.xticks(size=fsize, weight='bold')
        else:
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
            plt.ylabel(strings[2], fontsize=fsize, fontweight=fweight)
            plt.yticks(size=fsize, weight='bold')
    cid = fig.canvas.mpl_connect('button_press_event', GetClick)


def PlotTest(t,sig):
# ------------------
# Debugger for plots
# ------------------
    fig = plt.figure()

    plt.plot(t,sig)

    fsize = 14
    cbarNorth = False
    PlotLabels(fig,['$t$ [s]','Amplitude [arb.]'],fsize,cbarNorth,[],[])

    plt.show()
    return


def ImageTest(t,f,dats):
# ------------------
# Debugger for plots
# ------------------
    fig, ax = plt.subplots()

    #im = ax.imshow(dats, cmap='plasma')
    im = ax.pcolor(t,f,dats, cmap='plasma', shading='auto')
    
    fsize = 14
    fweight = 'normal'
    cbarNorth = False
    PlotLabels(fig,['$t$ [s]','$f$ [Hz]','$b^2(f_1,f_2)$'],fsize,cbarNorth,ax,im)

    plt.show()
    return


def SignalGen(fS,tend,Ax,fx,Afx,Ay,fy,Afy,Az,Ff,noisy):
# ------------------
# Provides FM test signal
# ------------------        
    t = np.arange(0,tend,1/fS)  # Time-vector sampled at "fS" Hz

    # Make 3 sinusoidal signals...
    dfx = Afx*np.sin(2*np.pi*t*Ff)  
    dfy = Afy*np.cos(2*np.pi*t*Ff)
    x = Ax*np.sin(2*np.pi*(fx*t + dfx))              # f1
    y = Ay*np.sin(2*np.pi*(fy*t + dfy))              # f2
    z = Az*np.sin(2*np.pi*(fx*t + fy*t + dfx + dfy)) # f1 + f2

    sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)
    return sig,t,fS


def ApplySTFT(sig,samprate,subint,step,nfreq,t0,detrend,errlim):
# ------------------
# STFT static method
# ------------------
    N = 1
    M = 1 + (len(sig) - subint)//step 
    lim  = nfreq//2                 # lim = |_ Nyquist/res _|
    time_vec = np.zeros(M)          # Time vector
    err  = np.zeros((N,M))          # Mean information
    spec = np.zeros((lim,M,N),dtype=complex)      # Spectrogram
    fft_coeffs = np.zeros((N,lim),dtype=complex)  # Coeffs for slice
    afft = np.zeros((N,lim))        # Coeffs for slice
    Ntoss = 0                       # Number of removed slices
    
    win = HannWindow(nfreq)         # Apply Hann window
    
    print(' Working...      ')
    for m in range(M):
        LoadBar(m,M)

        time_vec[m] = t0 + m*step/samprate
        for k in range(N):
            Ym = sig[m*step : m*step + subint] # Select subinterval 
            #Ym = sig[k,0:subint-1 + m*step] # Select subinterval     
            Ym = Ym[0:nfreq]            # Take only what is needed for res
            if detrend:                 # Remove linear least-squares fit
                Ym = ApplyDetrend(Ym)
            mean = sum(Ym)/len(Ym)
            Ym = win*(Ym-mean)  # Remove DC offset, multiply by window

            DFT = np.fft.fft(Ym)/nfreq  # Limit and normalize by vector length

            fft_coeffs[k,0:lim] = DFT[0:lim]  # Get interested parties
            dumft    = abs(fft_coeffs[k,:])       # Dummy for abs(coeffs)
            err[k,m] = sum(dumft)/len(dumft)     # Mean of PSD slice

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
    
    CWT = np.zeros((nyq,nyq),dtype=complex)

    fft_sig = np.fft.fft(sig)
    fft_sig = fft_sig[0:nyq]

    # Morlet wavelet in frequency space
    Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma/a) * np.exp( -2 * np.pi**2 * sigma**2 * ( freq_vec/a - f0)**2 )

    print(' Working...      ')
    for a in range(nyq):
        LoadBar(a,nyq)
        # Apply for each scale (read: frequency)
        CWT[a,:] = np.fft.ifft(fft_sig * Psi(a+1))
    print('\b\b\b^]\n')

    time_vec = np.arange(0,Nsig,2)/samprate
    return CWT,freq_vec,time_vec


def SpecToBispec(spec,v,lilguy):
# ------------------
# Turns spectrogram to b^2
# ------------------
    dum = spec.shape
    nfreq  = dum[0]
    slices = dum[1]

    lim = nfreq

    B  = np.zeros((lim//2,lim),dtype=complex)
    b2 = np.zeros((lim//2,lim))
    
    print('Calculating bicoherence...      ')     
    for j in range(lim//2):
        LoadBar(j,lim//2)
        
        for k in np.arange(j,lim-j):
            p1 = spec[k,:,v[0]]
            p2 = spec[j,:,v[1]]
            s  = spec[j+k,:,v[2]]

            Bi  = p1*p2*np.conj(s)
            e12 = abs(p1*p2)**2
            e3  = abs(s)**2

            Bjk = sum(Bi)                
            E12 = sum(e12)             
            E3  = sum(e3)                      

            b2[j,k] = (abs(Bjk)**2)/(E12*E3+lilguy) 

            B[j,k] = Bjk
    B = B/slices
    print ('\b\b^]\n')    
    return b2,B              


def SpecToCrossBispec(spec,v,lilguy):
# ------------------
# Turns 2 or 3 spectrograms to b^2
# ------------------
    dum = spec.shape
    nfreq  = dum[0]
    slices = dum[1]

    vec = np.arange(-(nfreq-1),nfreq)
    lim = 2*nfreq-1

    B  = np.zeros((lim,lim),dtype=complex)
    b2 = np.zeros((lim,lim))
    
    print('Calculating cross-bicoherence...      ')     
    for j in vec:
        LoadBar(j+nfreq,lim)           
        for k in vec:
            if abs(j+k) < nfreq:
                #p1 = (k>=0)*spec[abs(k),:,v[0]] + (k<0)*np.conj(spec[abs(k),:,v[0]])
                #p2 = (j>=0)*spec[abs(j),:,v[1]] + (j<0)*np.conj(spec[abs(j),:,v[1]])
                #s  = (j+k>=0)*spec[abs(j+k),:,v[2]] + (j+k<0)*np.conj(spec[abs(j+k),:,v[2]])

                p1 = np.real( spec[abs(k),:,v[0]] ) + 1j*np.sign(k)*np.imag( spec[abs(k),:,v[0]] )
                p2 = np.real( spec[abs(j),:,v[1]] ) + 1j*np.sign(j)*np.imag( spec[abs(j),:,v[1]] )
                s  = np.real( spec[abs(j+k),:,v[2]] ) + 1j*np.sign(j+k)*np.imag( spec[abs(j+k),:,v[2]] )

                Bi  = p1*p2*np.conj(s)
                e12 = abs(p1*p2)**2   
                e3  = abs(s)**2  

                Bjk = sum(Bi)                    
                E12 = sum(e12)             
                E3  = sum(e3)                     

                b2[j+nfreq-1,k+nfreq-1] = (abs(Bjk)**2)/(E12*E3+lilguy)

                B[j+nfreq-1,k+nfreq-1] = Bjk

    B = B/slices
    print('\b\b^]\n')                    
    return b2,B


def GetBispec(spec,v,lilguy,j,k,rando):
# ------------------
# Calculates the bicoherence of a single (f1,f2) value
# ------------------

    #p1 = spec[k,:,v[0]]
    #p2 = spec[j,:,v[1]]
    #s  = spec[j+k,:,v[2]]

    p1 = np.real( spec[abs(k),:,v[0]] ) + 1j*np.sign(k)*np.imag( spec[abs(k),:,v[0]] )
    p2 = np.real( spec[abs(j),:,v[1]] ) + 1j*np.sign(j)*np.imag( spec[abs(j),:,v[1]] )
    s  = np.real( spec[abs(j+k),:,v[2]] ) + 1j*np.sign(j+k)*np.imag( spec[abs(j+k),:,v[2]] )

    if rando:
        p1 = abs(p1)*np.exp[2j*np.pi*(2*np.rand(np.size(p1)) - 1)]
        p2 = abs(p2)*np.exp[2j*np.pi*(2*np.rand(np.size(p2)) - 1)]
        s  = abs(s)* np.exp[2j*np.pi*(2*np.rand(np.size(s)) - 1)]

    Bi  = p1*p2*np.conj(s)
    e12 = abs(p1*p2)**2
    e3  = abs(s)**2

    B   = sum(Bi)                 
    E12 = sum(e12)            
    E3  = sum(e3)                      

    w = (abs(B)**2)/(E12*E3+lilguy)
    
    B = B/len(Bi)
    return w,B,Bi


def HannWindow(N):
# ------------------
# Hann window
# ------------------
    win = (np.sin(np.pi*np.arange(N)/(N-1)))**2
    return win


def ApplyDetrend(y):
# ------------------
# Remove linear trend
# ------------------
    n = len(y)
    dumx  = np.arange(1,n+1) 
    s = (6/(n*(n**2-1)))*(2*sum(dumx*y) - sum(y)*(n+1))
    y = y - s*dumx
    return y


def ScaleToString(scale):
# ------------------
# Time/freq scaling
# ------------------
    tags = ['n',[],[],'$\mu$',[],[],'m','c','d','', [],[],'k',[],[],'M',[],[],'G',[],[],'T']
    s = tags[9+scale]
    return s   


def LoadBar(m,M):
# ------------------
# Help the user out!
# ------------------
    ch1 = '||\-/|||'
    ch2 = '_.:"^":.'
    buf = '\b\b\b\b\b\b\b%3.0f%%%s%s' % (100*m/(M-1), ch1[m%8], ch2[m%8])
    print(buf)
    return


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


def RunDemo():
# ------------------
# Demonstration
# ------------------
    b = BicAn
    x, t, fS = SignalGen(200,100,1,22,5,1,45,15,1,1/20,0.5)
    spec, afft, freq_vec, time_vec, err, Ntoss = ApplySTFT(x,fS,512,128,512,0,True,1e15)
    b2, B = SpecToBispec(spec,[0,0,0],1e-6)

    b.Processed = x
    b.SampRate  = fS
    b.sg = spec
    b.ft = afft
    b.fv = freq_vec
    b.tv = time_vec
    b.er = err
    b.bc = b2
    b.bs = B
    b.TScale = 3
    b.FScale = -1

    b.PlotSpectro(b)
    return b
