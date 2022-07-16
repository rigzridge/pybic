#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#      ______     ______ _      
#      | ___ \    | ___ (_)          Bicoherence Analysis Module for Python
#      | |_/ /   _| |_/ /_  ___      --------------------------------------
#      |  __/ | | | ___ \ |/ __|           
#      | |  | |_| | |_/ / | (__      v1.0 (c) 2022 -- G. Riggs | T. Matheny          
#      \_|   \__, \____/|_|\___|              
#             __/ |                      WVU Dept. of Physics & Astronomy
#            |___/                       
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
# inData    -> time-series [numpy.array (N,)/(N,1)/(N,2)/(N,3) & transpose]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# additional options... (see below for instructions)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# autoscale -> autoscaling in figures                  [default :: False]
# bispectro -> computes bispectrogram                  x[default :: False]
# cbarnorth -> control bolorbar location               [default :: True]
# cmap      -> adjust colormap                         [default :: 'viridis']
# dealias   -> applies antialiasing (LP) filter        x[default :: False]
# detrend   -> remove linear trend from data           [default :: False]
# errlim    -> mean(fft) condition                     [default :: 1e15] 
# filter    -> xxxxxxxxxxxxxxx                         x[default :: 'none']
# freqres   -> desired frequency resolution [Hz]       [default :: 0]
# fscale    -> scale for plotting frequencies          [default :: 0]
# justspec  -> true for just spectrogram               [default :: False]
# lilguy    -> set epsilon                             [default :: 1e-6]
# note      -> optional string for documentation       [default :: ' '] 
# plotit    -> start plotting tool when done           [default :: False]
# plottype  -> set desired plottable                   [default :: 'bicoh']
# samprate  -> sampling rate in Hz                     [default :: 1]
# sigma     -> parameter for wavelet spectrum          [default :: 0]
# spectype  -> set desired time-freq. method           [default :: 'stft']
# step      -> step size for Welch method in samples   [default :: 512]
# subint    -> subinterval size in samples             [default :: 128]
# sizewarn  -> warning for matrix size                 x[default :: True]
# smooth    -> smooths FFT by n samples                x[default :: 1]
# tscale    -> scale for plotting time                 [default :: 0]
# tzero     -> initial time                            [default :: 0]
# verbose   -> allow printing of info structure        [default :: False]
# window    -> select window function                  x[default :: 'hann']
# zpad      -> add zero-padding to end of time-series  [default :: False]
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Version History
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/15/2022 -> Fixed concatenation bug in ApplyZPad(), SignalGen() now outputs
# (1,N) numpy arrays instead of (N,), and lambda has been moved out of loop 
# in ApplyCWT(). Working on Colab notebook exposition. Switched everything to 
# column vectors, so even my first comment today is wrong! I know it's kind of
# brutal but it's sensible in Python. Allows x[0:n] instead of x[:,0:n] stuff!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/14/2022 -> Working on GUI!!! Reading a bunch of Tkinter documentation and 
# I think I know how to proceed, but gimme a few and I'll let you know. [...]
# So, I think that Tkinter is the way, but I was kind of confused earlier b/c
# I was using the built-in matplotlib widgets (which I think are back-ended 
# with Tkinter) to switch colormaps, etc. At this point, I don't think that 
# the extra headache -- however small -- is worth it right now, so I'll stick
# with the widgets. We have bells now, version 2.0 can have whistles!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/13/2022 -> Added SpecToCoherence() and Coherence() methods. [...] Adding
# support for np.loadtxt(...) stuff with a FileDialog(). Will need a try block
# in the future, but for now it's actually best to try to force some errors!
# I tried to get an exception with transposes of various inputs, but it seems
# like all is well. =^o Also: ran a cross-wavelet-bicoherence analysis(!), and
# fixed issue with TestSignal('cross_circle') that prevented viable b^2.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/12/2022 -> Figured out the weight='bold' on ticklabels in subplots! Here
# are my notes from before it though: {Clearly there's some way to do it, but 
# we need the effect of plt.xticks(size=..., weight=...) for axes (ax_i), and 
# ax.set_xticks(ticks, labels, **kwargs) isn't exactly the same! According to
# the docs, you can only pass text params if "label" has been supplied, else
# ax.tick_params(...) is required. I've tried copying lab = ax.get_ticklabels() 
# first, but ax.set_ticklabels(lab) does nothing. I scanned the source for the 
# wrapper function that _must_ exist, but it crossed my eyes a bit.} [...] The
# "trick" was just thinking about how I'd brute-force the problem in Matlab
# (i.e., setting the axes before labeling)! Also, we finally have cross-b^2 
# support! Technically, we had bug-tested the routine, but never sent in 2D 
# arrays of 2 or 3 time-series. All inData now parsed as np.array, so... bugs?
# Changed SpectroWavelet() to reflect recent changes with Matlab version
# (can finally handle cross-analysis!), everything seems benchmarked. $^/
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/11/2022 -> Merged main branch with a patch from Tyler; between the both
# of us, we're just about done with the necessary stuff! Slight debugging.
# Added automatic option for "Sigma" parameter; now using plt.tight_layout()
# to prevent that annoying "my axes labels exceed the figure" thing! Added
# PlotPowerSpec() method, and adjusted PlotLabels() for GUI template.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/10/2022 -> Adjusted things in ParseInput() method to not confuse Python
# with "self = BicAn" calls inside loop. This all started when I noticed
# bugs with bic.BicAn('demo') stuff... if self.RunBicAn was set to False 
# _before_ the "self =" assignment, the ProcessData() loop wouldn't start,
# but setting it False after the assignment left the RunBicAn property True
# for the original object. My idea: Assigning self to a function's output
# just instantiated _another_ object, instead of copying over the input. 
# Kind of fun to think about, but pernicious as all hell. Rewrote it to 
# avoid such nonsense -> demo inputs now set data & ParseInput() again.
# Changed pcolor to pcolormesh (documentation says it's faster!), figured 
# out how to overplot lines and such... No "hold on/off" nonsense needed.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/09/2022 -> Hoping to finish input parsing. [...] Sweet! Constructor is 
# all but finished, and "ParseInput" method is done... Changed the input 
# routine again to be case insensitive (looks like the Matlab approach). 
# Tyler knocked out a few more methods, input options changed slightly
# to allow string ('input', 'demo', etc.) as only input, cleaned up 
# constructor a bit, added "TestSignal" and rudimentary "ProcessData".
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/08/2022 -> Cleaned up a couple things, discovered the ternary operator
# equivalent of the C magic "cond ? a : b" => "a if cond else b". Passed out
# early so I kind of missed the night shift. %^/
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/07/2022 -> Tyler tackled the tedium of porting static methods over from
# the Matlab version. Bit of debugging, but things are all but error-free.
# Fiddling with plot methods, font sizes, colorbar locations, etc. 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/05/2022 --> Fixed some issues with STFT method; first "tests" attempted.
# Added GetClick method to obtain mouse coordinates on click ~> should be
# incredibly helpful down the road when we're trying to get the GUI up.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/04/2022 --> Copy pasta'd some code from MATLAB class.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/01/2022 --> First "code." 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# THINGS TO DO!
# **Swap out matplotlib widgets for full tkinter GUI =^x
# ** Figure out setter functions
# ** Configure warnings
# *_ Implement some kind of check for Raw data! Should eliminate string, etc.

# Methods left:
#{
# (1) Required
# ...ProcessData

# (2) Extra but nice
# SpecToCoherence
# Coherence
# PlotPointOut

# (3) More to learn about Python...
# SwitchPlot
# ...PlotGUI
# RefreshGUI
# MakeMovie
# etc.
#}

# Import dependencies
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from math import isclose
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# Extra, definitely-not-bug-tested stuff
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *

# Define classes for bispec script

class BicAn:
# Bicoherence analysis class for DSP
    
    # Attributes
    WarnSize  = 1024
    Date      = datetime.now()
    MaxRes    = 0
    Samples   = 0
    NFreq     = 0

    # Private attributes
    _RunBicAn  = False
    _NormToNyq = False
    _Nseries   = 1
    _WinVec    = []     

    Note      = ' '
    Raw       = []
    Processed = []
    History   = ' '

    SampRate  = 1
    FreqRes   = 0
    SubInt    = 512
    Step      = 128
    Window    = 'hann'       
    Sigma     = 0
    JustSpec  = False
    SpecType  = 'stft'
    Bispectro = False

    ErrLim    = 1e15
    FScale    = 0
    TScale    = 0
    Filter    = 'none' 
    Smooth    = 1  
    LilGuy    = 1e-6
    SizeWarn  = True
    BicVec    = [0,0,0]

    PlotIt    = True
    CMap      = 'viridis'
    CbarNorth = True
    PlotType  = 'bicoh'
    ScaleAxes = 'manual'
    LineWidth = 2
    FontSize  = 20
    PlotSlice = 0
    PlotSig   = 0

    Verbose   = False
    Detrend   = False
    ZPad      = False
    Cross     = False
    Vector    = False
    TZero     = 0

    Figure    = 0
    AxHands   = [0, 0, 0]

    tv = [] # Time vector
    fv = [] # Frequency vector
    ff = [] # Full frequency vector
    ft = [] # Fourier amplitudes
    sg = [] # Spectrogram (complex)
    cs = [] # Cross-spectrum
    cc = [] # Cross-coherence
    sg = [] # Coherence spectrum
    bs = [] # Bispectrum
    bc = [] # Bicoherence spectrum
    bp = [] # Biphase proxy
    bg = [] # Bispectrogram
    er = [] # Mean & std dev of FFT
    mb = [] # Mean b^2
    sb = [] # Std dev of b^2


    # Methods
    def __init__(self,inData,**kwargs):
    # ------------------
    # Constructor
    # ------------------ 
        self.ParseInput(inData,kwargs)

        if self._RunBicAn:
            self.ProcessData()

        return

    # Dependent properties
    @property
    def MaxRes(self):  # Maximum resolution
        return self.SampRate / self.SubInt

    @property
    def NFreq(self):   # Number of Fourier bins
        return int(self.SampRate / self.FreqRes)

    @property
    def _Nseries(self): # Number of time series
        return min(self.Raw.shape)

    @property
    def Samples(self): # Samples in data
        #val = len(self.Raw) if len(self.Processed)==0 else len(self.Processed)
        val = max(self.Raw.shape) if len(self.Processed)==0 else max(self.Processed.shape)
        return val

    # Set functions
    # @Raw.setter
    # def Raw(self, Raw):
    #      self.__Raw = Raw
    #      return


    def ParseInput(self,inData,kwargs):
    # ------------------
    # Handle inputs
    # ------------------
        self._RunBicAn = True  
        print('Checking inputs...') 

        if len(kwargs)==0:
            if isinstance(inData,type(BicAn)):
                # If object, output or go to GUI
                # HAS BUGS!
                print('Input is BicAn! ')
                #self = inData 
                self._RunBicAn = False

            elif isinstance(inData,list):
                # If array input, use normalized frequencies
                self.Raw       = inData
                self.FreqRes   = 1/self.SubInt    
                self._NormToNyq = True

            elif isinstance(inData,str):
                # Check string inputs
                self._RunBicAn = False
                dum = inData.lower()

                #### Should this be global?
                siglist = ['demo','classic','tone','noisy','2tone','3tone','line','circle','fast_circle','cross_2tone','cross_3tone','cross_circle']
                if dum == 'input':
                    # Start getfile prompt
                    infile = FileDialog()
                    # Try comma-separated? ### BUGS!
                    sig = np.loadtxt(infile, delimiter=',')  

                    self.ParseInput(sig,{}) 

                elif dum in siglist:
                    # If explicit test signal (or demo), confirm with user, then recursively call ParseInputs
                    dum = 'circle' if dum == 'demo' else dum
                    if messagebox.askokcancel('Question','Run the "{}" demo?'.format(dum)):
                        sig,_,fS = TestSignal(dum)
                        self.ParseInput(sig,{'SampRate':fS})  
                else:
                    print('Hmmm. That string isn`t supported yet... Try "demo".')   

            else:
                print('***WARNING*** :: Input must be BicAn object, array, or valid option! "{}" class is not supported.'.format(type(inData)))
        else:
            
            sz = inData.shape
            # Check if 1 or 2D numpy array
            if len(sz)<3 and isinstance( inData, type(np.array(0)) ):

                N = max(sz)                     # Get long dimension
                if len(sz)==1:                  # Check vector
                    self.Raw = np.zeros((N,1))  # Initialize array
                    self.Raw[:,0] = inData      # Place data

                elif len(sz)==2:                      # Must be 2D
                    self.Raw = np.zeros(N,(min(sz)))  # Initialize
                    if sz[1] > sz[0]:                 # Check row vector
                        inData = np.transpose(inData) # Transpose if so

                    self.Raw = inData                 # Boom!

            else:
                error()

            for key, val in kwargs.items():          # Loop through all keyword : value pairs

                # There are 2 ways to do this...
                # The first approach is somewhat simpler, but precludes case insensitivity =^\ 
                # try:                               # Throws error if input isn't a valid attribute
                #     dum = getattr(self, key)       # Get attribute
                #     if isinstance(val,type(dum)):  # Check type 
                #         setattr(self, key, val)    # Set!
                #     else: 
                #         print('***WARNING*** :: {} must be a {}! Using default value = {}'.format(key,type(dum),dum))
                # except AttributeError:
                #     print('***WARNING*** :: BicAn has no {} attribute!'.format(key))

                # This is how it's coded in the Matlab version, and doesn't care about cases! (Slower though...)
                dum  = dir(BicAn)                                       # Get class info as list of strings
                attr = [x.lower() for x in dum if x[0] != "_"]          # Keep only attributes, make lowercase
                if key.lower() in attr:                                 # Make input lowercase
                    for k in range(len(attr)):                          # Loop through all attributes (slow part!)
                        if key.lower() == attr[k]:                      # If input is an attribute:
                            dumval = eval( 'self.{}'.format(dum[k]) )   # Get default value for type comparison
                            if isinstance(val, type(dumval)):           # Check type
                                setattr(self, dum[k], val)              # Set attribute
                            else: 
                                print('***WARNING*** :: {} must be a {}! Using default value = {}'.format(dum[k],type(dumval),dumval))        
                else:
                    print('***WARNING*** :: BicAn has no {} attribute!'.format(key))

            # These input checks must be done in this order!
            self.SubInt = int(abs(self.SubInt))            # Remove sign and decimals
            if self.SubInt==0 or self.SubInt>self.Samples: # Check subinterval <= total samples
                self.SubInt = min(512,self.Samples)        # Choose 512 as long as data isn't too short
                print('***WARNING*** :: Subinterval too large for time-series... Using {}.'.format(self.SubInt))

            self.FreqRes = abs(self.FreqRes)               # Remove sign
            if self.FreqRes==0:                            # Check max res option
               self.FreqRes = self.MaxRes                  # Maximum resolution  
            elif self.FreqRes<self.MaxRes or self.FreqRes>self.SampRate/2:
                print('***WARNING*** :: Requested resolution not possible, using maximum ({} Hz).'.format(self.MaxRes))
                self.FreqRes = self.MaxRes

            if self.NFreq>self.SubInt:                     # Check if Fourier bins exceed subinterval
                print('***WARNING*** :: Subinterval too small for requested resolution... Using required.')
                self.FreqRes = self.MaxRes                 # Really hate repeating code, but...   

            self.Step = int(abs(self.Step))                # Remove sign and decimals
            if self.Step==0 or self.Step>self.SubInt:      # Check step <= subinterval
                self.Step = self.SubInt//4                 # This seems fine?
                print('***WARNING*** :: Step must be nonzero and less than subint... Using {}.'.format(self.Step))     

        print('done.')
        return


    def ApplyZPad(self):
    # ------------------
    # Zero-padding
    # ------------------
        if self.ZPad:
            tail_error = self.Samples % self.SubInt
            if tail_error != 0:
                # Add enough zeros to make subint evenly divide samples

                dum = np.zeros((self.SubInt-tail_error, self._Nseries))
                self.Processed = np.concatenate( (self.Raw, dum ) )
            else:
                self.Processed = self.Raw
        else:
            # Truncate time series to fit integer number of stepped subintervals
            samplim = self.Step * (self.Samples - self.SubInt) // self.Step + self.SubInt
            self.Processed = self.Raw[0:samplim]


    def ProcessData(self):
    # ------------------
    # Main processing loop
    # ------------------
        start = time.time()

        self.ApplyZPad()
        dum = self.SpecType.lower()
        if dum in ['fft', 'stft', 'fourier']:
            self.SpectroSTFT()
            self.SpecType = 'stft'
        elif dum in ['wave', 'wavelet', 'cwt']:
            self.SpectroWavelet()
            self.SpecType = 'wave'   

        if self.Cross:
            self.Coherence()

        if not self.JustSpec:
            self.Bicoherence()

        ##################
        end = time.time()

        buf = 'Complete! Process required %.5f s.' % (end-start)
        print(buf)

        if self.Verbose:
            print(self)      

        if self.PlotIt:       
            self.PlotGUI()


    ## Analysis
    def SpectroSTFT(self):
    # ------------------
    # STFT method
    # ------------------ 
        if self.NFreq>self.WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.NFreq)

        spec,afft,f,t,err,Ntoss = ApplySTFT(self.Processed,self.SampRate,self.SubInt,self.Step,self.NFreq,self.TZero,self.Detrend,self.ErrLim)
        
        self.tv = t
        self.fv = f

        self.ft = afft

        self.sg = spec
        self.er = err     
        return  


    def SpectroWavelet(self):
    # ------------------
    # Wavelet method
    # ------------------
        if self.Sigma == 0: # Check auto
            self.Sigma = 5*self.Samples/self.SampRate

        if self.Detrend:
            for k in range(self._Nseries):
                self.Processed[:,k] = ApplyDetrend(self.Processed[:,k])

        # Subtract mean
        for k in range(self._Nseries):
            self.Processed[:,k] = self.Processed[:,k] - sum(self.Processed[:,k]) / len(self.Processed[:,k]) 
        
        # Warn prompt
        if self.Samples>self.WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.Samples)

        CWT,acwt,f,t = ApplyCWT(self.Processed,self.SampRate,self.Sigma)

        self.tv = t + self.TZero
        self.fv = f
        self.ft = acwt 
        self.sg = CWT
        return


    def Coherence(self):
    # ------------------
    # Cross-spectrum/coh
    # ------------------
        if self._Nseries!=2:
            print('***WARNING*** :: Cross-coherence requires exactly 2 signals!')
        else:
            cspec,crosscoh,coh = SpecToCoherence(self.sg,self.LilGuy)
            self.cs = cspec      # Cross-spectrum
            self.cc = crosscoh   # Cross-coherence
            self.cg = coh        # Cohero-gram
        return


    def Bicoherence(self):
    # ------------------
    # Calculate bicoherence
    # ------------------       
        dum = self.sg 
        if self.SpecType == 'wave':
            WTrim = 50*2
            dum = self.sg[:,WTrim:-WTrim,:] 
        if self._Nseries==1:
            self.BicVec = [0, 0, 0]
            b2,B = SpecToBispec(dum,self.BicVec,self.LilGuy)
        else:
            if self._Nseries==2:
                self.BicVec = [0, 1, 1]
            else:
                self.BicVec = [0, 1, 2]
            b2,B = SpecToCrossBispec(dum,self.BicVec,self.LilGuy)
            self.ff = np.concatenate((-self.fv[::-1], self.fv[1::]))

        self.bs = B
        self.bc = b2
        return


    def CalcMean(self,Ntrials):
    # ------------------
    # Calculate mean of b^2
    # ------------------
        n,m,r = self.sg.shape

        A = abs(self.sg)
        eps = 1e-16
                
        self.mb = np.zeros( (self.bc.shape) )
        self.sb = np.zeros( (self.bc.shape) )

        for k in range(Ntrials):

            P = np.exp( 2j*np.pi * (2*np.random.random((n,m,r)) - 1) )

            if self._Nseries==1:
                dumspec,_ = SpecToBispec(A*P,self.BicVec,self.LilGuy)
            else:
                dumspec,_ = SpecToCrossBispec(A*P,self.BicVec,self.LilGuy)
            old_est   = self.mb/(k + eps) # "eps" is just a convenience for first loop, since mb = 0 initially       
                    
            self.mb += dumspec
            # "Online" algorithm for variance 
            self.sb += (dumspec - old_est)*(dumspec - self.mb/(k+1))
    
        self.mb /= Ntrials
        self.sb /= (Ntrials-1)
        return  


    ## Plot methods
    def PlotPowerSpec(self,*args):
    # ------------------
    # Plot power spectrum
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots()
        else:
            fig = args[0]
            ax  = args[1]

        f = self.fv/10**self.FScale

        for k in range(self._Nseries):
            ax.semilogy(f,self.ft[:,k]**2,linewidth=self.LineWidth)
        fstr = r'$f$ [%sHz]' % (ScaleToString(self.FScale))
        ystr = r'$|\mathcal{%s}|^2$ [arb.]' % ('P' if self.SpecType=='stft' else 'W')
        PlotLabels(fig,[fstr,ystr],self.FontSize,self.CbarNorth,ax,None)
        ax.set_xlim(f[0], f[-1])
        #plt.grid(True)

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotSpectro(self,*args):
    # ------------------
    # Plot spectrograms
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots()
        else:
            fig = args[0]
            ax  = args[1]

        tstr = r'Time [%ss]' % (ScaleToString(self.TScale))
        fstr = r'$f$ [%sHz]' % (ScaleToString(self.FScale))
        cbarstr = r'$\log_{10}|\mathcal{%s}(t,f)|^2$' % ('P' if self.SpecType=='stft' else 'W')

        t = self.tv/10**self.TScale
        f = self.fv/10**self.FScale

        im = ax.pcolormesh(t,f,2*np.log10(abs(self.sg[:,:,self.PlotSig])), cmap=self.CMap, shading='auto')
        PlotLabels(fig,[tstr,fstr,cbarstr],self.FontSize,self.CbarNorth,ax,im)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(f[0], f[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotBispec(self,*args):
    # ------------------
    # Plot bispectrum
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots()
        else:
            fig = args[0]
            ax  = args[1]

        dum, cbarstr = self.WhichPlot()

        if self._Nseries==1:
            f = self.fv/10**self.FScale
            im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto')
            ax.set_ylim(f[0], f[-1]/2)

            # Draw triangle
            ax.plot([0, f[-1]/2],[0, f[-1]/2],     color=[0.5,0.5,0.5], linewidth=2.5)
            ax.plot([f[-1]/2, f[-1]],[f[-1]/2, 0], color=[0.5,0.5,0.5], linewidth=2.5)

        else:
            f = self.ff/10**self.FScale
            im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto')
            ax.set_ylim(f[0], f[-1])
        
        fstr1 = r'$f_1$ [%sHz]' % (ScaleToString(self.FScale))
        fstr2 = r'$f_2$ [%sHz]' % (ScaleToString(self.FScale))
        PlotLabels(fig,[fstr1,fstr2,cbarstr],self.FontSize,self.CbarNorth,ax,im)
        ax.set_xlim(f[0], f[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return      


    def WhichPlot(self):
    # ------------------
    # Helper method for plots
    # ------------------
        guy = self.PlotType
        if guy == 'bicoh':
            dum = self.bc
            cbarstr = r'$b^2(f_1,f_2)$'
        elif guy in ['abs','real','imag','angle']:
            dum = eval('np.{}(self.bs)'.format(guy))
            cbarstr = r'%s%s $\mathcal{B}(f_1,f_2)$' % (guy[0].upper(),guy[1:].lower())
            cbarstr = r'$\beta(f_1,f_2)$' if guy=='angle' else cbarstr
        elif guy == 'mean':
            dum = self.mb
            cbarstr = r'$\langle b^2(f_1,f_2)\rangle$'
        elif guy == 'std':
            dum = self.sb
            cbarstr = r'$\sigma_{b^2}(f_1,f_2)$'
        return dum,cbarstr


    def PlotConfidence(self):

    # Needs debugged!

    # ------------------
    # Plot confidence interval
    # ------------------
        old_plot = self.PlotType
        old_dats = self.bc
        self.PlotType = 'bicoh'
        noise_floor   = -self.mb*np.log(1-0.999)
        #self.bc       =  self.bc * (self.bc>noise_floor)
        self.bc       = noise_floor
        #############
        self.bc = noise_floor
        self.PlotBispec()
        self.bc       = old_dats
        self.PlotType = old_plot


    def RefreshGUI(self):
    # ------------------
    # GUI test
    # ------------------ 
        fig = self.Figure

        # ax1 = self.AxHands[0]
        # ax2 = self.AxHands[1]
        # ax3 = self.AxHands[2]

        for k in range(3):
            self.AxHands[k].clear()
        
        self.PlotSpectro(fig,self.AxHands[1])
        self.PlotPowerSpec(fig,self.AxHands[2])
        self.PlotBispec(fig,self.AxHands[0])

        plt.tight_layout()
        plt.show()
        return


    def PlotGUI(self):
    # ------------------
    # GUI test
    # ------------------
        fig = plt.figure()

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(224)

        # Save figure and axes with object
        self.Figure = fig
        self.AxHands = [ax1, ax2, ax3]
    
        # # This is very primitive GUI stuff
        # ####
        # axcolor = [0.9, 0.9, 0.9]
        # rax1 = plt.axes([0.0, 0.0, 0.15, 0.1], facecolor=axcolor)
        # radio1 = RadioButtons(rax1, ['PiYG', 'viridis', 'gnuplot2'], active=0)
        # radio1.on_clicked(self.ChangeCMap)

        # rax2 = plt.axes([0.6, 0.85, 0.05, 0.1], facecolor=axcolor)
        # radio2 = RadioButtons(rax2, ['1', '2', '3'], active=0)
        # radio2.on_clicked(self.DumFunc)
        # ####
        
        self.RefreshGUI()
        return


    def DumFunc(self,event):
        if int(event) <= self._Nseries:
            self.PlotSig = int(event) - 1
        self.RefreshGUI()
        return

    def ChangeCMap(self,event):
        self.CMap = event
        self.RefreshGUI()
        return


    def SizeWarnPrompt(self,n):
    # ------------------
    # Prompt for CPU health
    # ------------------
        qwer = messagebox.askokcancel('Question','FFT elements exceed {}! ({}) Continue?'.format(self.WarnSize,n))
        if not qwer:
            print('Operation terminated by user.')
            self._RunBicAn = False
        return         # Bail if that seems scary! 


# Module methods

def FileDialog():
# ------------------
# Ganked from StackExchange...
# ------------------
    root = tk.Tk()
    # Build a list of tuples for each file type the file dialog should display
    my_ftypes = [('all files', '.*'), ('text files', '.txt'), ('dat files', '.dat')]
    # Ask the user to select a single file name
    ans = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select a file:",filetypes=my_ftypes)  
    return ans


def PlotLabels(fig,strings,fsize,cbarNorth,ax,im):
# ------------------
# Convenience function
# ------------------
    #plt.rcParams["font.weight"] = "bold"
    # Uncomment this for ALL bold
    plt.sca(ax)       # YESSSSSS!
    n = len(strings)
    fweight = 'normal'
    tickweight = 'bold'

    ax.set_xlabel(strings[0], fontsize=fsize, fontweight=fweight)
    if n>1:
        ax.set_ylabel(strings[1], fontsize=fsize, fontweight=fweight)
    plt.xticks(size=fsize, weight=tickweight)
    plt.yticks(size=fsize, weight=tickweight)
    ax.minorticks_on()
    if n>2:
        divider = make_axes_locatable(ax)
        if cbarNorth:
            cax = divider.append_axes('top', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='horizontal')   
            cax.xaxis.set_label_position('top') 
            cax.xaxis.tick_top()     
            cax.set_xlabel(strings[2], fontsize=fsize, fontweight=fweight)
            plt.xticks(size=fsize, weight=tickweight)
        else:
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
            cax.set_ylabel(strings[2], fontsize=fsize, fontweight=fweight)
            plt.yticks(size=fsize, weight=tickweight)
    cid = fig.canvas.mpl_connect('button_press_event', GetClick)
    return


def PlotTest(t,sig):
# ------------------
# Debugger for plots
# ------------------
    fig, ax = plt.subplots()

    plt.plot(t,sig)

    fsize = 14
    cbarNorth = False
    PlotLabels(fig,['$t$ [s]','Amplitude [arb.]'],fsize,cbarNorth,ax,None)

    plt.tight_layout()
    plt.show()
    return


def ImageTest(t,f,dats):
# ------------------
# Debugger for plots
# ------------------
    fig, ax = plt.subplots()

    #im = ax.imshow(dats, cmap='plasma')
    im = ax.pcolormesh(t,f,dats, cmap='plasma', shading='auto')
    
    fsize = 14
    fweight = 'normal'
    cbarNorth = False
    PlotLabels(fig,['$t$ [s]','$f$ [Hz]','$b^2(f_1,f_2)$'],fsize,cbarNorth,ax,im)

    plt.tight_layout()
    plt.show()
    return


def SignalGen(fS,tend,Ax,fx,Afx,Ay,fy,Afy,Az,Ff,noisy):
# ------------------
# Provides 3-osc FM test signal
# ------------------
# [sig,t] = SignalGen(fS,tend,Ax,fx,Afx,Ay,fy,Afy,Az,Ff,noisy)
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
# INPUTS:
# fS    --> Sampling frequency in Hz
# tend  --> End time [t = 0:1/fS:tend]
# Ax    --> Amplitude of oscillation #1
# fx    --> Frequency "       "      #1
# Afx   --> Amplitude of frequency sweep
# Ay    --> Amplitude of oscillation #2
# fy    --> Frequency "       "      #2
# Afy   --> Amplitude of frequency sweep
# Az    --> Amplitude of oscillation #3
# Ff    --> Frequency of frequency mod.
# noisy --> Noise floor
# OUTPUTS:
# sig   --> Signal 
# t     --> Time vector
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        
    t = np.arange(0,tend,1/fS)  # Time-vector sampled at "fS" Hz

    # Make 3 sinusoidal signals...
    dfx = Afx*np.sin(2*np.pi*t*Ff)  
    dfy = Afy*np.cos(2*np.pi*t*Ff)
    x = Ax*np.sin(2*np.pi*(fx*t + dfx))              # f1
    y = Ay*np.sin(2*np.pi*(fy*t + dfy))              # f2
    z = Az*np.sin(2*np.pi*(fx*t + fy*t + dfx + dfy)) # f1 + f2

    sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)
    #sig = np.reshape(sig, (1,len(sig))) # Output 1xN numpy array
    return sig,t,fS


def TestSignal(whatsig):
# ------------------
# Provides FM test signal
# ------------------
    fS   = 200
    tend = 100
    noisy = 2
    dum = whatsig.lower()
    if dum == 'classic':
        inData,t,_ = SignalGen(fS,tend,1,45,6,1,22,10,1,1/20,noisy)
    elif dum == 'tone':
        inData,t,_ = SignalGen(fS,tend,1,22,0,0,0,0,0,0,noisy)
    elif dum == 'noisy':
        inData,t,_ = SignalGen(fS,tend,1,22,0,0,0,0,0,0,5*noisy)
    elif dum == '2tone':
        inData,t,_ = SignalGen(fS,tend,1,22,0,1,45,0,0,0,noisy)
    elif dum == '3tone':
        inData,t,_ = SignalGen(fS,tend,1,22,0,1,45,0,1,0,noisy)
    elif dum == 'line':
        inData,t,_ = SignalGen(fS,tend,1,22,0,1,45,10,1,1/20,noisy)
    elif dum == 'circle':
        inData,t,_ = SignalGen(fS,tend,1,22,10,1,45,10,1,1/20,noisy)
    elif dum == 'fast_circle':
        inData,t,_ = SignalGen(fS,tend,1,22,5,1,45,5,1,5/20,noisy)
    elif dum == 'cross_2tone':
        x,t,_  = SignalGen(fS,tend,1,22,0,0,0,0,0,0,noisy)
        y,_,_  = SignalGen(fS,tend,1,45,0,0,0,0,0,0,noisy)
        inData = np.zeros( (len(t), 2) )
        inData[:,0] = x 
        inData[:,1] = x + y 
    elif dum == 'cross_3tone':
        x,t,_  = SignalGen(fS,tend,1,22,0,0,0,0,0,0,noisy)
        y,_,_  = SignalGen(fS,tend,1,45,0,0,0,0,0,0,noisy)
        z,_,_  = SignalGen(fS,tend,1,67,0,0,0,0,0,0,noisy)
        inData = np.zeros( (len(t), 3) )
        inData[:,0] = x 
        inData[:,1] = y 
        inData[:,2] = z
    elif dum == 'cross_circle':
        x,t,_  = SignalGen(fS,tend,1,22,10,0,0,0,0,1/20,noisy)
        y,_,_  = SignalGen(fS,tend,0,0 ,0 ,1,45,10,0,1/20,noisy)
        z,_,_  = SignalGen(fS,tend,0,22,10,0,45,10,1,1/20,noisy)
        inData = np.zeros( (len(t), 3) )
        inData[:,0] = x 
        inData[:,1] = y 
        inData[:,2] = z
    else:
        print('***WARNING*** :: "{}" test signal unknown... Using single tone..'.format(self.whatsig)) 
        inData,t,fS = SignalGen(fS,tend,1,22,0,0,0,0,0,0,0)
    return inData,t,fS


def ApplySTFT(sig,samprate,subint,step,nfreq,t0,detrend,errlim):
# ------------------
# STFT static method
# ------------------
    N = min(sig.shape)
    M = 1 + (max(sig.shape) - subint)//step
    lim  = nfreq//2                 # lim = |_ Nyquist/res _|
    time_vec = np.zeros(M)          # Time vector
    err  = np.zeros((M,N))          # Mean information
    spec = np.zeros((lim,M,N),dtype=complex)      # Spectrogram
    fft_coeffs = np.zeros((lim,N),dtype=complex)  # Coeffs for slice
    afft = np.zeros((lim,N))        # Coeffs for slice
    Ntoss = 0                       # Number of removed slices
    
    win = HannWindow(nfreq)         # Apply Hann window
    
    print('Applying STFT...      ')
    for m in range(M):
        LoadBar(m,M)

        time_vec[m] = t0 + m*step/samprate
        for k in range(N):
            Ym = sig[m*step : m*step + subint, k] # Select subinterval    
            Ym = Ym[0:nfreq]            # Take only what is needed for res
            if detrend:                 # Remove linear least-squares fit
                Ym = ApplyDetrend(Ym)
            mean = sum(Ym)/len(Ym)
            Ym = win*(Ym-mean)  # Remove DC offset, multiply by window

            DFT = np.fft.fft(Ym)/nfreq  # Limit and normalize by vector length

            fft_coeffs[0:lim,k] = DFT[0:lim]  # Get interested parties
            dumft    = abs(fft_coeffs[:,k])       # Dummy for abs(coeffs)
            err[m,k] = sum(dumft)/len(dumft)     # Mean of PSD slice

            if err[m,k]>=errlim:
                fft_coeffs[:,k] = 0*fft_coeffs[:,k] # Blank if mean excessive
                Ntoss += 1

            afft[:,k]  += dumft                   # Welch's PSD
            spec[:,m,k] = fft_coeffs[:,k]         # Build spectrogram

    print('\b\b\b^]\n')
    
    freq_vec = np.arange(nfreq)*samprate/nfreq
    freq_vec = freq_vec[0:lim] 
    afft /= M     
    return spec,afft,freq_vec,time_vec,err,Ntoss


def ApplyCWT(sig,samprate,sigma):
# ------------------
# Wavelet static method
# ------------------
    N,Nsig = sig.shape
    nyq    = Nsig//2

    f0 = samprate/Nsig
    freq_vec = np.arange(nyq)*f0
    
    acwt = np.zeros((N,nyq))
    CWT  = np.zeros((nyq,nyq,N),dtype=complex)

    # Morlet wavelet in frequency space
    Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma/a) * np.exp( -2 * np.pi**2 * sigma**2 * ( freq_vec/a - f0)**2 )

    for k in range(N):
        fft_sig = np.fft.fft(sig[:,k])
        fft_sig = fft_sig[0:nyq]

        print('Applying CWT...      ')
        for a in range(nyq):
            LoadBar(a,nyq)
            # Apply for each scale (read: frequency)
            dum = np.fft.ifft(fft_sig * Psi(a+1))
            CWT[a,:,k] = dum

            acwt[k,a]  = sum(abs(dum)) / len(dum)
        print('\b\b\b^]\n')

    time_vec = np.arange(0,Nsig,2)/samprate
    return CWT,acwt,freq_vec,time_vec


def SpecToBispec(spec,v,lilguy):
# ------------------
# Turns spectrogram to b^2
# ------------------
    nfreq,slices,_ = spec.shape

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
    print ('\b\b\b^]\n')    
    return b2,B              


def SpecToCrossBispec(spec,v,lilguy):
# ------------------
# Turns 2 or 3 spectrograms to b^2
# ------------------
    nfreq,slices,_ = spec.shape

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

                p1 = np.real( spec[abs(k),:,v[0]] )   + 1j*np.sign(k)*np.imag( spec[abs(k),:,v[0]] )
                p2 = np.real( spec[abs(j),:,v[1]] )   + 1j*np.sign(j)*np.imag( spec[abs(j),:,v[1]] )
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
    print('\b\b\b^]\n')                    
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
        p1 = abs(p1)*np.exp[ 2j*np.pi* (2*np.random.random(np.size(p1)) - 1) ]
        p2 = abs(p2)*np.exp[ 2j*np.pi* (2*np.random.random(np.size(p2)) - 1) ]
        s  = abs(s)* np.exp[ 2j*np.pi* (2*np.random.random(np.size(s)) - 1) ]

    Bi  = p1*p2*np.conj(s)
    e12 = abs(p1*p2)**2
    e3  = abs(s)**2

    B   = sum(Bi)                 
    E12 = sum(e12)            
    E3  = sum(e3)                      

    w = (abs(B)**2)/(E12*E3+lilguy)
    
    B = B/len(Bi)
    return w,B,Bi


def SpecToCoherence(spec,lilguy):
# ------------------
# Cross-spectrum, cross-coherence, coherogram
# ------------------
    print('Calculating cross-coherence...')     
    ncol = spec.shape[1]

    C  = np.conj(spec[:,:,0]) * spec[:,:,1];
    N1 = sum( np.transpose( abs(spec[:,:,0])**2 ) ) / ncol
    N2 = sum( np.transpose( abs(spec[:,:,1])**2 ) ) / ncol
    
    cc = abs( sum( np.transpose(C) )/ ncol )**2
    cc = cc / (N1*N2)

    xx = (abs(C)**2) / ( ( abs(spec[:,:,0])**2 ) * ( abs(spec[:,:,1])**2 ) + lilguy )
    return C,cc,xx 


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
    buf = '\b\b\b\b\b\b\b%3.0f%%%s%s' % (100*(m+1)/M, ch1[m%8], ch2[m%8])
    # Changed m/(M-1) to (m+1)/M to avoid /0 errors 
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
    b = BicAn('circle')

    return b
