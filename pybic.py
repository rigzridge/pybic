#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#      ______     ______ _      
#      | ___ \    | ___ (_)          Bicoherence Analysis Module for Python
#      | |_/ /   _| |_/ /_  ___      --------------------------------------
#      |  __/ | | | ___ \ |/ __|     
#      | |  | |_| | |_/ / | (__                [ v0.9 ] - 2022
#      \_|   \__, \____/|_|\___|             
#             __/ |                          G. Riggs | T. Matheny
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
# note      -> optional string for documentation [default :: ' '] 
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
# 7/12/2022 -> Merged main branch with a patch from Tyler; between the both
# of us, we're just about done with the necessary stuff! Slight debugging.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/11/2022 -> Adjusted things in ParseInput() method to not confuse Python
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
# 7/10/2022 -> Tyler knocked out a few more methods, input options changed
# slightly to allow string ('input', 'demo', etc.) as only input, cleaned 
# up constructor a bit, added "TestSignal" and rudimentary "ProcessData".
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/09/2022 -> Hoping to finish input parsing. [...] Sweet! Constructor is 
# all but finished, and "ParseInput" method is done... Changed the input 
# routine again to be case insensitive (looks like the Matlab approach). 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/08/2022 -> Cleaned up a couple things, discovered the ternary operator
# equivalent of the C magic "cond ? a : b" => "a if cond else b". Passed out
# early so I kind of missed the night shift %^/
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/07/2022 -> Tyler tackled the tedium of porting static methods over from
# the Matlab version. Bit of debugging, but things are all but error-free.
# Fiddling with plot methods, font sizes, colorbar locations, etc. 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/05/2022 --> Fixed some issues with STFT method; first "tests" attempted.
# Added GetClick method to obtain mouse coordinates on click ~> should be
# incredibly helpful down the road when we're trying to get the GUI up.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/04/2022 --> Copy pasta'd some code from MATLAB class
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/01/2022 --> First "code." 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# THINGS TO DO!
# ** Configure warnings
# ** Figure out setter functions
# *_ Why doesn't "RunDemo" output proper object?
# *_ Clean up constructor
# ** Tackle input type/dimension dilemma [len() tests lists, not np arrays; arrays are not uniform!]
# ** Implement some kind of check for Raw data! Should eliminate string, etc.

# Methods left:
#{
# (1) Required
# PlotPowerSpect
# ...ProcessData

# (2) Extra but nice
# SpecToCoherence
# Coherence
# PlotPointOut

# (3) More to learn about Python...
# SwitchPlot
# PlotGUI
# RefreshGUI
# MakeMovie
# etc.
#}

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from math import isclose
import time


# Define classes for bispec script

class BicAn:
# Bicoherence analysis class for DSP
    
    # Properties
    FontSize  = 20
    WarnSize  = 1024
    Date      = datetime.now()
    MaxRes    = 0
    Samples   = 0
    NFreq     = 0
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
    def __init__(self,inData,**kwargs):
    # ------------------
    # Constructor
    # ------------------ 
        self.ParseInput(inData,kwargs)

        if self.RunBicAn:
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
    def Samples(self): # Samples in data
        val = len(self.Raw) if len(self.Processed)==0 else len(self.Processed)
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
        self.RunBicAn = True  
        print('Checking inputs...') 

        if len(kwargs)==0:
            if isinstance(inData,type(BicAn)):
                # If object, output or go to GUI
                # HAS BUGS!
                print('Input is BicAn! ')
                #self = inData 
                self.RunBicAn = False

            elif isinstance(inData,list):
                # If array input, use normalized frequencies
                self.Raw       = inData
                self.SampRate  = 1 
                self.FreqRes   = 1/self.SubInt    
                self.NormToNyq = True

            elif isinstance(inData,str):
                # Check string inputs
                self.RunBicAn = False
                dum = inData.lower()

                #### Should this be global?
                siglist = ['classic','tone','noisy','2tone','3tone','line','circle','cross_2tone','cross_3tone','cross_circle']
                if dum == 'input':
                    # Start getfile prompt
                    ans = FileDialog()

                    # {PARSE FILE...}

                elif dum in siglist or dum == 'demo':
                    # If explicit test signal (or demo), confirm with user, then recursively call ParseInputs
                    ans = 'Test signal!'
                    dum = 'tone' if dum == 'demo' else dum
                    if messagebox.askokcancel('Question','Run the "{}" demo?'.format(dum)):
                        sig,_,fS = TestSignal(dum)
                        self.ParseInput(sig,{'SampRate':fS})  
                else:
                    ans = 'Hmmm. That string isn`t supported yet... Try ''demo'''
                print(ans)

            else:
                print('***WARNING*** :: Input must be BicAn object, array, or valid option! "{}" class is not supported.'.format(type(inData)))
        else:
            
            self.Raw = inData

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


    def ProcessData(self):
    # ------------------
    # Main processing loop
    # ------------------

        # {SOMETHING TO MAKE .PROCESSED ARRAY...?}

        start = time.time()

        self.ApplyZPad()
        dum = self.SpecType.lower()
        if dum in ['fft', 'stft', 'fourier']:
            self.SpectroSTFT()
            self.SpecType = 'stft'
        elif dum in ['wave', 'wavelet', 'cwt']:
            self.SpectroWavelet()
            self.SpecType = 'wave'    

        if not self.JustSpec:
            self.Bicoherence()

        ##################
        end = time.time()

        buf = 'Processing complete! Execution required %3.3f s.' % (end-start)
        print(buf)

        self.PlotSpectro()
        self.PlotBispec()


    def SpectroSTFT(self):
    # ------------------
    # STFT method
    # ------------------ 
        if self.NFreq>self.WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.NFreq)

        spec,afft,f,t,err,Ntoss = ApplySTFT(self.Processed,self.SampRate,self.SubInt,self.Step,self.NFreq,self.TZero,self.Detrend,self.ErrLim)
        
        self.tv = t
        self.fv = f

        # for k in range(self.Nseries):
        #     self.ft[k,:] = mean(abs(spec(:,:,k)'))
        self.ft = afft

        self.sg = spec
        self.er = err     
        return  


    def SpectroWavelet(self):
    # ------------------
    # Wavelet method
    # ------------------
        if self.Detrend:
            self.Processed = ApplyDetrend(self.Processed)
        # Subtract mean
        self.Processed = self.Processed - sum(self.Processed)/len(self.Processed)
        
        # Warn prompt
        if self.Samples>self.WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.Samples)

        nyq = self.Samples//2
        CWT = np.zeros((nyq,nyq,self.Nseries),dtype=complex)

        for k in range(self.Nseries):
            CWT[:,:,k],f,t = ApplyCWT(self.Processed,self.SampRate,self.Sigma)

        self.tv = t
        self.fv = f
        #self.ft =     
        self.sg = CWT
        return


    def Bicoherence(self):
    # ------------------
    # Calculate bicoherence
    # ------------------       
        dum = self.sg 
        if self.SpecType == 'wave':
            WTrim = 50*2
            dum = self.sg[:,WTrim:end-WTrim,:] 
        if self.Nseries==1:
            v = [0, 0, 0]
            b2,B = SpecToBispec(dum,v,self.LilGuy)
        else:
            if self.Nseries==2:
                v = [0, 1, 1]
            else:
                v = [0, 1, 2]
            b2,B = SpecToCrossBispec(dum,v,self.LilGuy)
            self.ff = np.concatenate((-self.fv[::-1], self.fv[1::]))

        self.bs = B
        self.bc = b2
        return


    def PlotSpectro(self):
    # ------------------
    # Plot spectrograms
    # ------------------
        tstr = 'Time [%ss]' % (ScaleToString(self.TScale))
        fstr = '$f$ [%sHz]' % (ScaleToString(self.FScale))
        dum  = 'P' if self.SpecType=='stft' else 'W'
        cbarstr = '$\log_{10}|\mathcal{%s}(t,f)|^2$' % (dum)

        fig, ax = plt.subplots()
        for k in range(self.Nseries):
            im = ax.pcolormesh(self.tv/10**self.TScale,self.fv/10**self.FScale,2*np.log10(abs(self.sg[:,:,k])), cmap=self.CMap, shading='auto')
            PlotLabels(fig,[tstr,fstr,cbarstr],self.FontSize,self.CbarNorth,ax,im)

        plt.show()
        return


    def PlotBispec(self):
    # ------------------
    # Plot bispectrum
    # ------------------
        fig, ax = plt.subplots()

        dum, cbarstr = self.WhichPlot()

        if self.Nseries==1:
            f = self.fv/10**self.FScale
            im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto')

            # Draw triangle
            plt.plot([0, f[-1]/2],[0, f[-1]/2],     color=[0.5,0.5,0.5], linewidth=2.5)
            plt.plot([f[-1]/2, f[-1]],[f[-1]/2, 0], color=[0.5,0.5,0.5], linewidth=2.5)

        else:
            f = self.ff/10**self.FScale
            im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto')
        
        fstr1 = '$f_1$ [%sHz]' % (ScaleToString(self.FScale))
        fstr2 = '$f_2$ [%sHz]' % (ScaleToString(self.FScale))
        PlotLabels(fig,[fstr1,fstr2,cbarstr],self.FontSize,self.CbarNorth,ax,im)

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
        elif guy == 'mean':
            dum = self.mb
            cbarstr = r'$\langle b^2(f_1,f_2)\rangle$'
        elif guy == 'std':
            dum = self.sb
            cbarstr = r'$\sigma_{b^2}(f_1,f_2)$'
        return dum,cbarstr


    def CalcMean(self,Ntrials):
    # ------------------
    # Calculate mean of b^2
    # ------------------
        dum = self.sg.shape
        n   = dum[0]
        m   = dum[1]
        r   = dum[2]

        v = [0,0,0]
        A = abs(self.sg)
        eps = 1e-16
                
        self.mb = np.zeros((n//2,n))
        self.sb = self.mb

        for k in range(Ntrials):

            P = np.exp( 2j*np.pi * (2*np.random.random((n,m,r)) - 1) )

            dumspec,_ = SpecToBispec(A*P,v,self.LilGuy)
            old_est   = self.mb/(k + eps) # "eps" is just a convenience for first loop, since mb = 0 initially       
                    
            self.mb += dumspec
            # "Online" algorithm for variance 
            self.sb += (dumspec - old_est)*(dumspec - self.mb/(k+1))
    
        self.mb /= Ntrials
        self.sb /= (Ntrials-1)
        return


    def PlotConfidence(self):

    # Needs debugged!

    # ------------------
    # Plot confidence interval
    # ------------------
        old_plot = self.PlotType
        old_dats = self.bc
        self.PlotType = 'bicoh'
        noise_floor   = -self.mb*np.log(1-0.999)
        self.bc       =  self.bc * (self.bc>noise_floor)
        #############
        self.bc = noise_floor
        self.PlotBispec
        self.bc       = old_dats
        self.PlotType = old_plot
            

    def ApplyZPad(self):

    # Needs debugged!

    # ------------------
    # Zero-padding
    # ------------------
        if self.ZPad:
            tail_error = self.Samples % self.SubInt
            if tail_error != 0:
                # Add enough zeros to make subint evenly divide samples
                
                #self.Processed = np.concatenate( self.Raw, np.zeros((self.Nseries, self.SubInt-tail_error)) )
                self.Processed = np.concatenate(( self.Raw, np.zeros(self.SubInt-tail_error) ))

            else:
                self.Processed = self.Raw
        else:
            # Truncate time series to fit integer number of stepped subintervals
            samplim = self.Step* (self.Samples - self.SubInt)//self.Step + self.SubInt
            
            #self.Processed = self.Raw[:,0:samplim]
            self.Processed = self.Raw[0:samplim]


    def SizeWarnPrompt(self,n):
    # ------------------
    # Prompt for CPU health
    # ------------------
        qwer = messagebox.askokcancel('Question','FFT elements exceed {}! ({}) Continue?'.format(self.WarnSize,n))
        if not qwer:
            print('Operation terminated by user.')
            self.RunBicAn = False
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
    im = ax.pcolormesh(t,f,dats, cmap='plasma', shading='auto')
    
    fsize = 14
    fweight = 'normal'
    cbarNorth = False
    PlotLabels(fig,['$t$ [s]','$f$ [Hz]','$b^2(f_1,f_2)$'],fsize,cbarNorth,ax,im)

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

    # Will have to FIX THIS later...
    #sig = np.zeros((1,len(t)))

    # Make 3 sinusoidal signals...
    dfx = Afx*np.sin(2*np.pi*t*Ff)  
    dfy = Afy*np.cos(2*np.pi*t*Ff)
    x = Ax*np.sin(2*np.pi*(fx*t + dfx))              # f1
    y = Ay*np.sin(2*np.pi*(fy*t + dfy))              # f2
    z = Az*np.sin(2*np.pi*(fx*t + fy*t + dfx + dfy)) # f1 + f2

    sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)
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
        x,t,_ = SignalGen(fS,tend,1,22,0,0,0,0,0,0,noisy)
        y,_,_ = SignalGen(fS,tend,1,45,0,0,0,0,0,0,noisy)
        inData[0,:] = x 
        inData[1,:] = y 
    elif dum == 'cross_3tone':
        x,t,_ = SignalGen(fS,tend,1,22,0,0,0,0,0,0,noisy)
        y,_,_ = SignalGen(fS,tend,1,45,0,0,0,0,0,0,noisy)
        z,_,_ = SignalGen(fS,tend,1,67,0,0,0,0,0,0,noisy)
        inData[0,:] = x 
        inData[1,:] = y 
        inData[2,:] = z
    elif dum == 'cross_circle':
        x,t,_ = SignalGen(fS,tend,1,22,10,0,0,0,0,1/20,noisy)
        y,_,_ = SignalGen(fS,tend,1,45,10,0,0,0,0,1/20,noisy)
        z,_,_ = SignalGen(fS,tend,0,22,10,0,45,10,1,1/20,noisy)
        inData[0,:] = x 
        inData[1,:] = y 
        inData[2,:] = z
    else:
        print('***WARNING*** :: "{}" test signal unknown... Using single tone..'.format(self.whatsig)) 
        inData,t,fS = SignalGen(fS,tend,1,22,0,0,0,0,0,0,0)
    return inData,t,fS


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
    
    print('Applying STFT...      ')
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

    print('Applying CWT...      ')
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
    print ('\b\b\b^]\n')    
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
