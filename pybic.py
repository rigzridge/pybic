import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define classes for bispec script

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


        plt.plot(t,sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [arb.]")
        plt.title("Test signal!")
        plt.show()

    def ImageTest(dats):
    # ------------------
    # Debugger for plots
    # ------------------

        fig, ax = plt.subplots()

        im = ax.imshow(dats, cmap='plasma')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        #plt.xlabel("Time [s]")
        #plt.ylabel("Amplitude [arb.]")
        #plt.title("Test signal!")

        #fig.set_size_inches(18, 16)
        ax.invert_yaxis()
        plt.show()

    def LoadBar(self,m,M):
        # ------------------
        # Help the user out!
        # ------------------

        ch1 = ['|','|','!','-','/','|','|','|'] 
        ch2 = ['_','.',':',"'",'^',"'",':','.']
        print ('\b\b\b\b\b\b\b%3.0f%%%s%s', (100*m/M, ch1[m%8], ch2[m%8]))
        return m,M
        
        
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
        return sig, t, fS


    def ApplySTFT(self,sig,samprate,subint,step,nfreq,t0,detrend,errlim):
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
        Ntoss = 0                      # Number of removed slices
        
        win = np.sin(np.pi*np.arange(nfreq)/(nfreq-1)) # Apply Hann window
        
        print(' Working...      ')
        for m in range(M):
            self.LoadBar(m,M-1)

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


    def ApplyCWT(self,sig,samprate,sigma):
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
            self.LoadBar(a,nyq-1)
            # Apply for each scale (read: frequency)
            CWT[a,:] = np.fft.ifft(fft_sig * Psi(a+1))
        print('\b\b\b^]\n')

        time_vec = np.arange(0,Nsig,2)/samprate
        return CWT,freq_vec,time_vec


    def detrend_y(y):
        y = scipy.detrend(y)
        # ------------------
        # Remove linear trend
        # ------------------
        print(' Applying detrend...') 
        nfreq = np.len(y)
        dumx = np.arange(1,nfreq+1) 
        dumxy = dumx*y
        s = (6/(nfreq*(nfreq**2-1)))*(2*dumxy.sum() - y.sum()*(nfreq+1))
        y = y - s*dumx
        print('done.\n')
        # ApplyDetrend
    
    
    def SpecToBispec(self,spec,v,lilguy):
    # ------------------
    # Turns spectrogram to b^2
    # ------------------
        [nfreq,slices] = np.size(spec)

        lim = nfreq

        B = np.zeros(np.floor(lim/2),lim)
        b2 = np.zeros(np.floor(lim/2),lim)
        
        print('Calculating bicoherence...      ')     
        for j in range(lim//2):
            self.LoadBar(j,np.floor(lim/2)+1)
            
            for k in range(lim-j):
                p1 = spec[k,:,v[0]]
                p2 = spec[j,:,v[1]]
                s  = spec[j+k,:,v[2]]

                Bi  = (p1)*p2*np.conj(s)
                e12 = abs(p1*p2)^2
                e3  = abs(s)^2

                Bjk = sum(Bi)                
                E12 = sum(e12)             
                E3  = sum(e3)                      

                b2[j,k] = (abs(Bjk)^2)/(E12*E3+lilguy) 

                B[j,k] = Bjk
            B = B/np.slices
            print ('\b\b^]\n')    
            return b2,B               
    # SpecToBispec
    
    
    
    def SpecToCrossBispec(self,spec,v,lilguy):
    # ------------------
    # Turns 2 or 3 spectrograms to b^2
    # ------------------
        [nfreq,slices] = np.size(spec)
        vec=[]
        np.array(vec[-nfreq:nfreq])
        lim = 2*nfreq-1

        B = np.zeros(lim,lim)
        b2 = np.zeros(lim,lim)
        
        print('Calculating cross-bicoherence...      ')     
        for j in vec:
            self.LoadBar(j+nfreq,lim)           
            for k in vec:
                if abs(j+k) < nfreq:
                    p1 = (k>=0)*spec[abs(k),:,v[0]] + (k<0)*np.conj[spec[abs(k),:,v[0]]]
                    p2 = (j>=0)*spec[abs(j),:,v[1]] + (j<0)*np.conj[spec[abs(j),:,v[1]]]
                    s  = (j+k>=0)*spec[abs(j+k),:,v[2]] + (j+k<0)*np.conj[spec[abs(j+k),:,v[2]]]

                    Bi  = p1*p2*np.conj(s)
                    e12 = abs(p1*p2)^2   
                    e3  = abs(s)^2  

                    Bjk = sum(Bi)                    
                    E12 = sum(e12)             
                    E3  = sum(e3)                     

                    b2[j+nfreq,k+nfreq] = (abs(Bjk)^2)/(E12*E3+lilguy)

                    B[j+nfreq,k+nfreq] = Bjk
        B = B/slices
        print('\b\b^]\n')                    
        return b2,B
    # SpecToCrossBispec
    
    
    def GetBispec(spec,v,lilguy,j,k,rando):
    # ------------------
    # Calculates the bicoherence of a single (f1,f2) value
    # ------------------

        p1 = spec[k,:,v[0]]
        p2 = spec[j,:,v[1]]
        s  = spec[j+k-1,:,v[2]]

        # Negatives
        #p2 = conj(spec(j,:))
        #s  = conj(spec(j+k-1,:))

        if rando:
            p1 = abs(p1)*np.exp[2j*np.pi*(2*np.rand(np.size(p1)) - 1)]
            p2 = abs(p2)*np.exp[2j*np.pi*(2*np.rand(np.size(p2)) - 1)]
            s  = abs(s)* np.exp[2j*np.pi*(2*np.rand(np.size(s)) - 1)]

        Bi  = p1*p2*np.conj(s)
        e12 = abs(p1*p2)^2
        e3  = abs(s)^2

        B   = sum(Bi)                 
        E12 = sum(e12)            
        E3  = sum(e3)                      

        w = (abs(B)^2)/(E12*E3+lilguy)
        
        B = B/np.length(Bi)
        return w,B,Bi
    

    def HannWindow(N):
    # ------------------
    # Hann window
    # ------------------
        win = (np.sin(np.pi*range(N)/(N-1)))^2
        return win
    # HannWindow
    
    
    def PlotLabels(strings,fsize,cbarNorth):
    # ------------------
    # Convenience function
    # ------------------
        n = np.length(strings)
        plt.xlabel("%sstrings{0}", fontsize="%dfsize", fontdict='bold')
        if n>1:
            plt.ylabel("%sstrings{1}", fontsize="%dfsize", fontdict='bold')
        if n>2:
            if cbarNorth:
                cbar = plt.colorbar('location','NorthOutside') 
                plt.xlabel(cbar, "%sstrings{2}", fontsize="%dfsize", fontdict='bold')
            else:
                cbar = plt.colorbar()
                plt.ylabel("%sstrings{2}", fontsize="%dfsize", fontdict='bold')
        plt.gca('YDir','normal','fontsize',fsize, 'xminortick','on', 'yminortick','on')
        #set(gca,'YDir','normal', YDir is crucial w/ @imagesc!'fontsize',fsize, 'xminortick','on', 'yminortick','on')
        # PlotLabels
    
    def ScaleToString(scale):
    # ------------------
    # Frequency scaling
    # ------------------
        tags = {'n',[],[],'\mu',[],[],'m','c','d','', [],[],'k',[],[],'M',[],[],'G',[],[],'T'} 
        s = tags[10+scale]
        return s
    # ScaleToString        
    
    
    def RunDemo():
    # ------------------
    # Demonstration
    # ------------------
        bic = BicAn
        [x,t,fS] = bic.TestSignal('circle')
        #N = 512*2
        N = np.length(t)
        x = x[:,1:N]
        dT = t(N)-t(1)
        bic = BicAn(x,'sigma',dT*5,'spectype','stft','sizewarn','samprate',fS,'justspec','plottype','bicoh')
        return bic
    # RunDemo