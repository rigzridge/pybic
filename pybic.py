#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#      ______     ______ _      
#      | ___ \    | ___ (_)          Bicoherence Analysis Module for Python
#      | |_/ /   _| |_/ /_  ___      --------------------------------------
#      |  __/ | | | ___ \ |/ __|               v2.0 (c) 2022-2023
#      | |  | |_| | |_/ / | (__                        
#      \_|   \__, \____/|_|\___|              G. Riggs & T. Matheny
#             __/ |                      
#            |___/                       WVU Dept. of Physics & Astronomy
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
# bispectro -> computes bispectrogram                 x[default :: False]
# calccoi   -> cone of influence                       [default :: False]
# cbarnorth -> control colorbar location               [default :: True]
# cmap      -> adjust colormap                         [default :: 'viridis']
# dealias   -> apply antialiasing (LP) filter         x[default :: False]
# detrend   -> remove linear trend from data           [default :: False]
# errlim    -> mean(fft) condition                     [default :: 1e15] 
# filter    -> apply band-pass filter                 x[default :: 'none']
# freqres   -> desired frequency resolution [Hz]       [default :: 0]
# fscale    -> scale for plotting frequencies          [default :: 0]
# justspec  -> true for just spectrogram               [default :: False]
# lilguy    -> set epsilon                             [default :: 1e-6]
# limfreq   -> limits fourier bins to nyq/limfreq      [default :: 2]
# note      -> optional string for documentation       [default :: ' '] 
# plotit    -> start plotting tool when done           [default :: False]
# plottype  -> set desired plottable                   [default :: 'bicoh']
# samprate  -> sampling rate in Hz                     [default :: 1]
# sigma     -> parameter for wavelet spectrum          [default :: 0]
# spectype  -> set desired time-freq. method           [default :: 'stft']
# step      -> step size for Welch method in samples   [default :: 512]
# subint    -> subinterval size in samples             [default :: 128]
# sizewarn  -> warning for matrix size                x[default :: True]
# smooth    -> smooths FFT by n samples               x[default :: 1]
# trispec   -> estimates trispectrum                   [default :: False]
# tscale    -> scale for plotting time                 [default :: 0]
# tzero     -> initial time                            [default :: 0]
# verbose   -> allow printing of info structure        [default :: False]
# window    -> select window function                 x[default :: 'hann']
# zpad      -> add zero-padding to end of time-series  [default :: False]
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Version History 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/14/2023 -> Finally added FindMaxInRange() class method and ApplyFilter()
# module function [latter needs debugged!]; NoXLabel variable now used for 
# PlotHelper() to make simpler "stacked" plots of biphase and |B| (say) 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/12/2023 -> Changed alpha in scatter for PlotTrispec(); ApplyZPad() now
# only used when SpecType='stft', fixed oversight with ZPad=False; thinking
# about switching over Sigma to units of 1/f0 = N\Delta t; added AlphaExp 
# attribute; fixed transcription error in ApplyCWT() [np.sqrt(.../alpha**)
# was wrong! changed to np.sqrt(...) * alpha**]; WTrim now takes 10% instead
# of 100 points from both sides; ParseInput() sped up w/ ".index()" method;
# plots of pdf now show both measured and analytic uncertainty
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/11/2023 -> Fixed issue with MonteCarloMax()'s use of NFreq attribute;
# shuffled around PlotPointOut() behavior when PlotType='hybrid'; sort() now
# automatically applied to output of nRandSumLessThanUnity()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/10/2023 -> Now using ax.fill_between() to plot uncertainty in biphase
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/04/2023 -> Found out about ax.axhline and ax.axvline methods, swapped 
# out a few instances of ax.plot(...); using ax.axvspan() to show uncertainty 
# in measured value for 'b2Prob' plots; GetPolySpec() now accepts cross with
# v=[0,0,1,...] input; line coloring error fixed [50+40*k] -> [(50+40*k) % 256]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/03/2023 -> Fixed issue with Tkinter dialog in Colab notebook; all main
# plot functions now use PlotDPI as dpi of figure; ClickPlot() draws lines
# on spectrogram and power spectrum, indicating selections in frequency;
# noise floor [see vanMilligen PRL (1995)] now plotted in 'b2Prob' for CWT, 
# alternate expression [see ElgarGuza IEEE (1988)] used for STFT
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/28/2023 -> Support for 'femto-', 'pico-', 'hecto-', 'peta-', and 'exa-' 
# now included; finally changed fS output of TestSignal() to float()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/24/2023 -> Added a few helpful defaults in SpecTo...(), ...Spec() methods,
# in particular, SpecVLim = [] (auto), and NormBic = False (auto)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/19/2023 -> Smoothed out edges in PlotPointOut() so that legends actually
# appear [switched single plt.legend() call with four ax<#>.legend() calls]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/18/2023 -> Fixed issue with plotting local (inst.) b^2 instead of local  
# bispectrum when PlotType~='bicoh' and BicOfTime=True, required additions to
# WhichPlot() for proper strings; right arrow action fixed; PlotTimeline()
# module method added to plot lines with smooth color change, used for phasor
# plots; massive rewrite to PlotpointOut() and friends... now using switchyard
# (ish) idea in a single function [PlotHelper(str) w/ str='b2Prob', 'BvsTime'
# or 'Phasor'], same thing might benefit PlotBispec(), PlotSpectro(), etc.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/17/2023 -> Now using LimFreq property to make low-frequency analysis
# simpler, see 'circle_oversample' option; added set methods for 'PlotType',
# 'SpecType','TScale','FScale' props, PlotSlice prop updated to allow GUI 
# options from Matlab (click spectrogram, etc); "vLim" input for PlotSpectro()
# controls colorbar limits [think caxis(vLim)]; BicOfTime prop toggles plots
# of "instantaneous" bicoherence spectrum, SHIFT + {t,x} toggles or resets;
# bunch of small edits in ApplySTFT() and ApplyCWT() required for LimFreq 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/16/2023 -> Added LineColor property from BicAn.m for adding line plots
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/03/2023 -> Added "intermittent" quadratic coupling ('d3dtest') option,
# changed some of the default labels ['\mathcal{P}(t,f)'->'X(t,f)'] for STFT,
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/01/2023 -> Fixed tiny issue with CWT (zeroth bin was f0, not 0)...
# Why do I have a memory of thinking about this??? Am I wrong now? Am I tired?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/27/2023 -> Developed and debugged PlotPhasor() method for normalized plots 
# of biphase, led to significant changes to PlotPointOut(), pulled out the 
# routines for plots and made new methods [Plotb2Prob(), PlotBvsTime()], added 
# 'hybrid' option for plotting [b^2(f1,f2) x biphase(f1,f2)], using ClickPlot()
# while PlotType='hybrid' gives subplots (will surely change!), fiddled with 
# Plotlabels() defaults to eliminate annoying repetition (yay Python!)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/20/2023 -> Fixed issue with reference in MonteCarloMax(); adjusted 'input'
# option to accommodate input dialog [resurrected FileDialog()!]; changed 
# default sigma to pi*(...) instead of 5*(...); lots of small fixes; finally 
# got tick issues figured out with list of labels; removed small matplotlib 
# toggles, now use SHIFT + {1,2,3} = {!,@,#} to switch for cross; read about 
# and implemented __setattr__ method to check on issues with case, etc; cone
# cone of influence (COI) added to wavelet scaleogram; added DrawSimplex()
# method to ease drawing of tricoherence domain; figured out issue with dum
# vars acting like pointers [see MonteCarloMax()]; added PlotTrispec() method;
# fixed issue in PlotPointOut() for cross-bispectra, changed some labels to 
# conform to LaTeX style; looking to add colormap selector in PlotGUI() [not
# done yet! Will have to get back to this...]; updated PlotLabels() method to 
# finally(!) fix issue with tick labels and ALSO annoying thing with colorbar
# labels resizing all the time; added defaults for SignalGen() so TestSignal() 
# is easier to read [actually have sone a bit of this kind of thing!], new
# test signals ('helix','3tone_short',&c.); PlotTrispec() allows user to color
# scatter with either tricoherence OR triphase [maybe automate this???]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/19/2023 -> Changed fonts to LaTeX (computer modern, 'cm'); added support
# for nth-order polyspectrum with GetPolySpec(...); included a few more test
# signals ('quad_couple','cube_couple','coherence',&c); migrated tricoherence 
# support from Matlab version; added local hill climb to Monte Carlo; added 
# flag to PlotPointOut() input to allow inputting freqs directly
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9/04/2022 -> Fixed bug with CaxHands not being refreshed by PlotGUI(),
# added a bit for limiting colorbar axes [think caxis(...)]-> still testing,
# fixed root window issue with SizeWarn, fiddled with nonlinear CWT scales,
# adjusted initialization to avoid issues with fractional SampRate
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/26/2022 -> Fixed annoying Tkinter root window thing with root.withdraw()
# and added keypress function -> some radiobuttons removed!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/25/2022 -> Sketched beta version of ClickPlot(), rough edges abound.
# Tyler's updates to bic.BicAn('input') dialog have been incorporated.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/24/2022 -> Debugged PlotPointOut() more completely, some issues remain 
# with multiple plots; still need to get legends figured out! 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/21/2022 -> Merged Tyler's addition of PlotPointOut(), slight edits for
# debugging. NewGUICax is now initialized.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/17/2022 -> Added CalcMean() button to emulate Matlab version; trying to 
# fix the issue with colorbar overplots! [...] Fixed with NewGUICax flag. 
# [...] Added PlotType radiobutton, and reverted SignalGen() output. _GR
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7/16/2022 -> Tyler here, reading up on GUI for python, Tkinter is the best
# option and can be implemented by importing tkinter alongside matplotlib. 
# Something like figure = plt.figure(stuff) ax = figure.add_subplot(numbers) 
# give it a type with FigureCanvasTKAgg(check documentation for options) then 
# needs tied up with a bow using thing.get_tk_widget().pack to actually put it 
# all together into your prompt widget, this isn't even a version history but
# idk how to comment... anyway
# tl;dr make an array or pandas data frame, feed it into tk canvas, pack it 
# together and decide on subplot layout (rows, cols, index)
# [...] WarnSize now private attribute; BicAn bails if inData is broken. _GR
# [...] Going to create a branch for Tkinter GUI. I resurrected the glitchy 
# code I'd tested a few days ago (7/14) and went from there. Comboboxes are 
# pretty sweet for colormaps (looking at you, BicAn 1.0!), so the switch to 
# Tkinter is probably worth it. The "postcommand" callback for comboboxes 
# seems promising to limit what options are given in the drop-down. Also,
# redundancy in PlotLabels() was fixed with a ternary. _GR
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

# THINGS TO DO! [** = not yet, *_ = kind of, __ = probably done]
# ** Add colormap picker in PlotGUI() with SHIFT + c, say
# *_ Swap out matplotlib widgets for full tkinter GUI =^x
# *_ Figure out setter functions
# ** Configure warnings
# __ Implement some kind of check for Raw data! Should eliminate string, etc.
# __ Fix colorbar axes overplotting each refresh
# __ Fix issue with colorbar labels when calling RefreshGUI()
# *_ Add buttons and callbacks from Matlab
# ** Swap out "dum" variables for more literate ones
# ** Comment the code!!!
# __ Fix butt-ugly inputs to PlotPointOut children!
# ** Flag for base units (maybe not based in time, say)
# ** Antialiased option!

# Methods left:
#{
# MakeMovie
# etc.
#}

# Import dependencies
import os
import time
# import warnings
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# Computer modern font (LaTeX default)
plt.rcParams['mathtext.fontset'] = 'cm'

# Define classes for bispec script

class BicAn:
# Bicoherence analysis class for DSP
    
    # Attributes
    Date      = datetime.now()
    MaxRes    = 0.
    Samples   = 0
    NFreq     = 0

    # Private attributes
    _WarnSize  = 1024
    _RunBicAn  = False
    _NormToNyq = False
    _Nseries   = 1
    _WinVec    = []     

    Note      = ' '
    Raw       = []
    Processed = []
    History   = ' '

    SampRate  = 1.
    FreqRes   = 0.
    SubInt    = 512
    Step      = 128
    LimFreq   = 2.
    Window    = 'hann'       
    Sigma     = 0.
    AlphaExp  = 0.5
    CalcCOI   = False
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
    TickLabel = 'normal'
    LineWidth = 2
    FontSize  = 20
    PlotDPI   = 150
    SpecVLim  = []
    NormBic   = False
    PlotSlice = None
    PlotSig   = 0
    BicOfTime = False

    Verbose   = False
    Detrend   = False
    ZPad      = False
    Cross     = False
    Trispec   = False
    Vector    = False
    TZero     = 0.

    Figure    = 0
    AxHands   = [0,0,0]
    CaxHands  = [None,None]
    NewGUICax = False

    tv = []   # Time vector
    fv = []   # Frequency vector
    ff = []   # Full frequency vector

    ft = []   # Fourier amplitudes
    sg = []   # Spectrogram (complex)

    cs = []   # Cross-spectrum
    cc = []   # Cross-coherence
    cg = []   # Coherence spectrum

    bs = []   # Bispectrum
    bc = []   # Bicoherence spectrum
    bp = []   # Biphase proxy
    bg = []   # Bispectrogram

    ts = []   # Trispectrum
    tc = []   # Tricoherence spectrum

    er = []   # Mean & std dev of FFT
    mb = []   # Mean b^2
    sb = []   # Std dev of b^2


    # Class methods
    def __init__(self,inData,**kwargs):
    # ------------------
    # Constructor
    # ------------------ 
        self.ParseInput(inData,kwargs)

        if self._RunBicAn:
            self.ProcessData()
        return

    def __setattr__(self, attr, val):
    # ------------------
    # Internal method to set attributes
    # ------------------
        if not attr in dir(BicAn):
            print('***WARNING*** :: BicAn class has no attribute {}!'.format(attr))
            # Check case issue
            dum_dir = dir(self)
            attrLow = attr.lower()
            lower_list = [dum.lower() for dum in dum_dir]
            if attrLow in lower_list:
                k = lower_list.index(attrLow)
                print('Did you mean {}?'.format(dum_dir[k]))

        else:
            # if isinstance(val, type( eval('self.{}'.format(attr)) ) ):
            #     print('Same class!')
            # else:
            #     print('Wrong class!')

            if attr in ['PlotType','SpecType','TScale','FScale']:
                if attr=='PlotType':
                    opts = ['bicoh','abs','real','imag','angle','mean','std','hybrid']
                elif attr=='SpecType':
                    opts = ['fft','stft','fourier','wave','wavelet','cwt']
                elif attr in ['TScale','FScale']:
                    opts = [-15,-12,-9,-6,-3,-2,-1,0,2,3,6,9,12,15,18]

                if val not in opts:
                    print('***WARNING*** :: Invalid input! "%s" does not correspond to an option.' % val)
                    print('Valid are:',opts)
                    # Keep current option
                    val = eval('self.%s' % attr)

            self.__dict__[attr] = val

            if attr=='SubInt':
                self.FreqRes = self.MaxRes
                print('***NOTE*** :: Resolution set to maximum!')
  
        return


    # Dependent properties
    @property
    def MaxRes(self):  # Maximum resolution
        return self.SampRate / self.SubInt if self.SpecType=='stft' else self.SampRate / self.Samples

    @property
    def NFreq(self):   # Number of Fourier bins  
        return int(self.SampRate / self.FreqRes / self.LimFreq)

    @property
    def _Nseries(self): # Number of time series
        return min(self.Raw.shape)

    @property
    def Samples(self): # Samples in data
        #val = len(self.Raw) if len(self.Processed)==0 else len(self.Processed)
        return max(self.Raw.shape) if len(self.Processed)==0 else max(self.Processed.shape)

    @property
    def LineColor(self): # For coloring plots
        return eval('cm.%s( np.linspace(0,1,256) )[:,0:3]' % self.CMap)


    def ParseInput(self,inData,kwargs):
    # ------------------
    # Handle inputs
    # ------------------
        self._RunBicAn = True  
        print('Checking inputs...') 

        if len(kwargs)==0:
            if isinstance(inData,np.ndarray):
                # If array input, use normalized frequencies
                self.Raw       = inData
                self.FreqRes   = 1/self.SubInt    
                self._NormToNyq = True
                self.ParseInput(inData,{'SampRate':1.})

            elif isinstance(inData,str):
                # Check string inputs
                self._RunBicAn = False
                instr = inData.lower()

                #### Should this be global?
                siglist = ['demo','classic','tone','noisy','2tone','3tone','4tone',
                            'line','circle','fast_circle','quad_couple','d3dtest','cube_couple',
                            'coherence','cross_2tone','cross_3tone','cross_circle','amtest','quad_couple_circle',
                            'quad_couple_circle2',
                            '3tone_short','circle_oversample','cross_3tone_short','helix']
                if instr == 'input':
                    # Start getfile prompt
                    # root = tk.Tk()
                    # root.withdraw()
                    # infile = filedialog.askopenfilename()
                    infile = FileDialog()

                    sig = np.loadtxt(infile) 
                    self.ParseInput(sig,{}) 

                elif instr in siglist:
                    # If explicit test signal (or demo), confirm with user, then recursively call ParseInputs
                    instr = 'circle' if instr == 'demo' else instr
                    try:
                        # Added for Colab notebooks, etc. No need for prompt there!
                        root = tk.Tk()
                        root.withdraw()
                        if messagebox.askokcancel('Question','Run the "{}" demo?'.format(instr), master=root):
                            sig,_,fS = TestSignal(instr)
                            self.ParseInput(sig,{'SampRate':fS})  
                        root.destroy()
                    except:
                        sig,_,fS = TestSignal(instr)
                        self.ParseInput(sig,{'SampRate':fS})

                else:
                    print('Hmmm. That string isn`t supported yet... Try "demo".') 
                    print(siglist)  

            else:
                print('***ERROR*** :: Input must be a numpy array or valid option! "{}" class is not supported.'.format(type(inData)))
                error()
        else:
            
            sz = inData.shape
            # Check if 1 or 2D numpy array
            if len(sz)<3 and isinstance( inData, np.ndarray ):

                N = max(sz)                     # Get long dimension
                if len(sz)==1:                  # Check vector
                    self.Raw = np.zeros((N,1))  # Initialize array
                    self.Raw[:,0] = inData      # Place data

                elif len(sz)==2:                      # Must be 2D
                    self.Raw = np.zeros((N,min(sz)))  # Initialize
                    if sz[1] > sz[0]:                 # Check row vector
                        inData = np.transpose(inData) # Transpose if so

                    self.Raw = inData                 # Boom!
                
                # For CWT, mostly
                self.Processed = self.Raw
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
                    k = attr.index(key.lower())                 # Get index
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
                self.Processed = self.Raw # Could remove this now!
        else:
            # Truncate time series to fit integer number of stepped subintervals
            samplim = self.Step * ((self.Samples - self.SubInt) // self.Step) + self.SubInt
            self.Processed = self.Raw[0:samplim]


    def ProcessData(self):
    # ------------------
    # Main processing loop
    # ------------------
        start = time.time()
 
        dum = self.SpecType.lower()
        if dum in ['fft', 'stft', 'fourier']:
            self.ApplyZPad()
            self.SpectroSTFT()
            self.SpecType = 'stft'
        elif dum in ['wave', 'wavelet', 'cwt']:
            self.SpectroWavelet()
            self.SpecType = 'wave'   

        if self.Cross:
            self.Coherence()

        if not self.JustSpec:
            self.Bicoherence()
            if self.Trispec:
                self.Tricoherence()

        ##################
        end = time.time()

        print('Complete! Process required %.5f s.' % (end-start))

        if self.Verbose:
            print(self)      

        if self.PlotIt and not self.JustSpec:      
            self.PlotGUI()


    ## Analysis
    def SpectroSTFT(self):
    # ------------------
    # STFT method
    # ------------------ 
        if self.NFreq>self._WarnSize and self.SizeWarn:
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
            #self.Sigma = np.pi*self.Samples/self.SampRate
            self.Sigma = np.pi

        if self.Detrend:
            for k in range(self._Nseries):
                self.Processed[:,k] = ApplyDetrend(self.Processed[:,k])

        # Subtract mean
        # print(self.Processed)
        for k in range(self._Nseries):
            self.Processed[:,k] = self.Processed[:,k] - sum(self.Processed[:,k]) / len(self.Processed[:,k]) 
        
        # Warn prompt
        if self.Samples>self._WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.Samples)

        CWT,acwt,f,t = ApplyCWT(self.Processed,self.SampRate,self.Sigma,self.LimFreq,self.AlphaExp)

        if self.CalcCOI:
            nz = np.zeros((len(self.Processed),1))
            nz[0,:] = 1
            nz[-1,:] = 1
            nzCWT,_,_,_ = ApplyCWT(nz,self.SampRate,self.Sigma,self.LimFreq,self.AlphaExp)
            for k in range(self._Nseries):
                coiMask = ( (abs(nzCWT)/np.max(abs(nzCWT)) ) < np.exp(-3) )
                CWT[:,:,k] = CWT[:,:,k] * coiMask[:,:,0]
                acwt[:,k]  = np.mean(abs(CWT[:,:,k])**2,1)

                #CWT[:,:,k] = nzCWT[:,:,k]

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
        if self.SpecType == 'wave' and not self.CalcCOI:
            #WTrim = 50*2
            WTrim = self.NFreq // 10
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


    def Tricoherence(self):
    # ------------------
    # Calculate tricoherence
    # ------------------       
        dum = self.sg 
        if self.SpecType == 'wave' and not self.CalcCOI:
            WTrim = 50*2
            dum = self.sg[:,WTrim:-WTrim,:] 
        if self._Nseries==1:
            self.BicVec = [0, 0, 0, 0]
            t2,T = SpecToTrispec(dum,self.BicVec,self.LilGuy)
        else:
            print('***WARNING*** :: Tricoherence currently only supports single time-series!')

        self.ts = T
        self.tc = t2
        return


    def CalcMean(self,Ntrials=10):
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


    def MonteCarloMax(self,N=2,Nrolls=1000,critCoh=1,plot=False,verbose=False):
    # ------------------
    # Toss some dice and try to find maxima!
    # ------------------ 
        start = time.time()

        bestCoh = 0

        flim = self.NFreq 

        vals = np.zeros(N)

        if plot:
            if N==2:
                plt.plot([0,flim/2],   [0,flim/2],color=[0.5,0.5,0.5], lw=2.5)
                plt.plot([flim/2,flim],[flim/2,0],color=[0.5,0.5,0.5], lw=2.5)
                plt.plot([0,flim],     [0,0],     color=[0.5,0.5,0.5], lw=2.5)
            elif N==3:
                ax = plt.figure().add_subplot(projection='3d')
                DrawSimplex(flim)

        for k in range(Nrolls):
            
            freqs = ( nRandSumLessThanUnity(N) * flim ).astype(int)

            nCoh,_,_ = GetPolySpec(self.sg, freqs, self.LilGuy)

            if verbose:
                print("Testing ", freqs, "nCoh = ", nCoh)

            if plot and nCoh>0.1:
                if N==2:
                    plt.plot(freqs[0],freqs[1],'o',color=[nCoh,0,nCoh])
                elif N==3:
                    ax.plot(freqs[0],freqs[1],freqs[2],'o',color=[nCoh,0,nCoh])

            if nCoh>bestCoh:
                bestCoh = nCoh
                bestFreqs = 1*freqs

                searchNeighbors = True if (min(bestFreqs)!=0 and max(bestFreqs)!=flim) else False

                cnt = 0
                while searchNeighbors:

                    cnt += 1
                    #if verbose:
                    print("Searching neighbors... %d" % cnt)

                    bestCoh_old = bestCoh
                    bestFreqs_old = 1*bestFreqs

                    for n in range(2*N):
                        # This is absolutely insane! Without the "1*" here, freqs acts like a pointer
                        freqs = 1*bestFreqs_old
                        freqs[n//2] += 1 if n%2==0 else -1
                        nCoh,_,_ = GetPolySpec(self.sg, freqs, self.LilGuy)
                        if nCoh>bestCoh:
                            bestCoh = nCoh
                            bestFreqs = 1*freqs

                    searchNeighbors = True if (min(bestFreqs)!=0 and max(bestFreqs)!=flim) else False
                        
                    if bestCoh==bestCoh_old:
                        searchNeighbors = False

                    if bestCoh>critCoh:
                        break

        if (N==2 or N==3) and plot:
            plt.show()
        print("Max found is nCoh = %.3f" % (bestCoh), " @ ", self.fv[bestFreqs]/10**self.FScale, "%sHz" % (ScaleToString(self.FScale)), "\nw/ indices ", bestFreqs)

        end = time.time()
        print('Complete! Process required %.5f s.' % (end-start))

        return bestCoh, bestFreqs


    ## Plot methods
    def PlotPowerSpec(self,*args,vLim=[]):
    # ------------------
    # Plot power spectrum
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
        else:
            fig = args[0]
            ax  = args[1]

        f = self.fv/10**self.FScale
        dum = self.ft if self.PlotSlice is None else abs(self.sg[:,self.PlotSlice,:])**2

        for k in range(self._Nseries):
            ax.semilogy(f, dum[:,k], linewidth=self.LineWidth, color=self.LineColor[(50+40*k) % 256])

        fstr = r'$f\,\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        ystr = r'$\langle|%s(f)|^2\rangle\,\mathrm{[arb.]}$' % ('X' if self.SpecType=='stft' else r'\mathcal{W}')
        PlotLabels(fig,ax,[fstr,ystr],self.FontSize,self.CbarNorth)
        ax.set_xlim(f[0], f[-1])
        if len(vLim)==2:
            ax.set_ylim(10**vLim[0],10**vLim[1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotSpectro(self,*args,vLim=[]):
    # ------------------
    # Plot spectrograms
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
            cax = None
        else:
            fig = args[0]
            ax  = args[1]
            cax = self.CaxHands[1]

        tstr = r'$t\, [\mathrm{%ss}]$' % (ScaleToString(self.TScale))
        fstr = r'$f\,\, [\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        cbarstr = r'$\log_{10}|%s(t,f)|^2$' % ('X' if self.SpecType=='stft' else r'\mathcal{W}')

        t = self.tv/10**self.TScale
        f = self.fv/10**self.FScale

        if len(vLim)==2:
            im = ax.pcolormesh(t,f,2*np.log10(abs(self.sg[:,:,self.PlotSig]) + 0*1e-16), cmap=self.CMap, shading='auto', vmin=vLim[0], vmax=vLim[1])
        else:
            im = ax.pcolormesh(t,f,2*np.log10(abs(self.sg[:,:,self.PlotSig]) + 0*1e-16), cmap=self.CMap, shading='auto')

        # cax = PlotLabels(fig,ax,[tstr,fstr,cbarstr],self.FontSize,self.CbarNorth,im,cax,cbarweight='ticks')
        cax = PlotLabels(fig,ax,[tstr,fstr,cbarstr],self.FontSize,self.CbarNorth,im,cax)
        if self.NewGUICax:
            self.CaxHands[1] = cax
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(f[0], f[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotBispec(self,*args,normb2=False,plot3d=False):
    # ------------------
    # Plot bispectrum
    # ------------------
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)

            if plot3d:
                ###
                #ax = fig.add_subplot(111, projection='3d')
                #ax = plt.figure().add_subplot(projection='3d')
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            cax = None
        else:
            fig = args[0]
            ax  = args[1]
            cax = self.CaxHands[0]

        if (self.PlotSlice is not None) and self.BicOfTime:
            n,_,m = self.sg.shape
            # 
            dumsg = self.sg[:,self.PlotSlice,:].reshape((n,1,m))
            
            if self._Nseries==1:
                b2,B = SpecToBispec(dumsg,self.BicVec,self.LilGuy)
            else:
                b2,B = SpecToCrossBispec(dumsg,self.BicVec,self.LilGuy)   

            dum,cbarstr = self.WhichPlot(local=B)
            dum = b2 if self.PlotType=='bicoh' else dum

        else:
            dum,cbarstr = self.WhichPlot()

        if self._Nseries==1:
            f = self.fv/10**self.FScale
            if plot3d:
                print('hey!')
                # EXPERIMENTAL 3D OPTION!!!
                X,Y = np.meshgrid(f, f[0:len(f)//2] )
                im = ax.plot_surface(X, Y, dum, cmap=self.CMap, lw=1, antialiased=True, rstride=10, cstride=10)
                return
            else:
                if normb2:
                    im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto', vmin=0, vmax=1)
                else:
                    im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto')

            ax.set_ylim(f[0], f[-1]/2)

            # Draw triangle
            ax.plot([0, f[-1]/2],[0, f[-1]/2],     color=[0.5,0.5,0.5], linewidth=2.5)
            ax.plot([f[-1]/2, f[-1]],[f[-1]/2, 0], color=[0.5,0.5,0.5], linewidth=2.5)

        else:
            f = self.ff/10**self.FScale
            if normb2:
                im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto', vmin=0, vmax=1)
            else:
                im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto')
            ax.set_ylim(f[0], f[-1])
        
        fstr1 = r'$f_1\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr2 = r'$f_2\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        cax = PlotLabels(fig,ax,[fstr1,fstr2,cbarstr],self.FontSize,self.CbarNorth,im,cax)
        if self.NewGUICax:
            self.CaxHands[0] = cax
        ax.set_xlim(f[0], f[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return      


    def PlotTrispec(self,Tval=0.5,colorTricoh=True):
    # ------------------
    # Plot trispectrum
    # ------------------

        f = self.fv / 10**self.FScale
        lim = len(f)
        lim2 = lim//2
        lim3 = lim//3

        max_t = np.max(self.tc)
        print('Max t^2 =',max_t)

        n = len(self.fv)
        X, Y, Z = np.meshgrid(self.fv[0:n], self.fv[0:n//2], self.fv[0:n//3])

        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()

        t = self.tc.flatten()
        T = self.ts.flatten()
        q = t>Tval
        #q = abs(t-Tval) < 0.1

        dum = t[q] if colorTricoh else np.angle(T[q])
        cbarstr = r'$t^2(f_1,f_2,f_3)$' if colorTricoh else r'$\gamma(f_1,f_2,f_3)$'

        #ax = plt.figure().add_subplot(projection='3d')
        #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        fig = plt.figure(dpi=self.PlotDPI)
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(x[q],y[q],z[q],c=dum,cmap=self.CMap,alpha=0.5)

        #isosurface(f(1:lim),f(1:lim2),f(1:lim3),bic.tc,Tval,angle(bic.ts))

        ax.set_xlim(0,f[-1]) 
        ax.set_ylim(0,f[-1]/2)
        ax.set_zlim(0,f[-1]/3)
        
        # d = {['Max::' num2str(max_t)];...
        #     ['Current::' num2str(Tval)]};
        # text(0.85,0.9, d ,...
        #              'units','normalized',...
        #              'color','black');

        fstr1 = r'$f_1\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr2 = r'$f_2\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr3 = r'$f_3\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        cax = PlotLabels(fig,ax,[fstr1,fstr2,fstr3,cbarstr],self.FontSize,self.CbarNorth,im,None)

        # divider = make_axes_locatable(ax)
        # cbarloc = 'top' if self.CbarNorth else 'right'
        # cax = divider.append_axes(cbarloc, size='5%', pad=0.05)
        #fig.colorbar(im,ax=ax,shrink=0.75)

        DrawSimplex(f[-1])
        
        plt.tight_layout()
        plt.show()
        
        return


    def WhichPlot(self,local=None):
    # ------------------
    # Helper method for plots
    # ------------------
        guy = self.PlotType
        if guy == 'bicoh':
            dum = self.bc
            cbarstr = r'$b^2(f_1,f_2)$'
        elif guy in ['abs','real','imag','angle']:
            if local is None:
                dum = eval('np.%s(self.bs)' % guy)
                Bstr = r'\mathcal{B}'
            else:
                dum = eval('np.%s(local)' % guy)
                Bstr = r'\widetilde{\mathcal{B}}'
            ###dum = eval('np.{}(self.bs)'.format(guy)) if local is None else eval('np.{}(local)'.format(guy))
            if guy in ['real','imag']:
                cbarstr = r'%s%s $%s(f_1,f_2)$' % (guy[0].upper(),guy[1:].lower(),Bstr)
            elif guy=='abs':
                cbarstr = r'$|%s(f_1,f_2)|$' % Bstr
            elif guy=='angle':
                cbarstr = r'$\beta(f_1,f_2)$'
        elif guy == 'mean':
            dum = self.mb
            cbarstr = r'$\langle b^2(f_1,f_2)\rangle$'
        elif guy == 'std':
            dum = self.sb
            cbarstr = r'$\sigma_{b^2}(f_1,f_2)$'
        elif guy == 'hybrid':
            dum = self.bc * np.angle(self.bs) / np.pi
            cbarstr = r'$b^2(f_1,f_2)\times\beta(f_1,f_2)$'
        else:
            dum = None
            cbarstr = ''
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


    def PlotHelper(self,whatPlot,X,Y,IsFreq=False,CheckNeighbors=False,fig=None,ax=None,Ntrials=200,b2bins=100,cVal=0.999,SaveAs=None,NoXLabel=False):
    # ------------------
    # Estimate and plot distribution of b2 for single point
    # ------------------

        doShow = False
        if fig is None:
            doShow = True
            fig, ax = plt.subplots(dpi=self.PlotDPI)

        # Okay, so ClickPlot() sends in X and Y w/ based on:
        ## self.fv if _Nseries==1
        ## self.ff if else
        fLocX = X
        fLocY = Y

        # Create dummy frequency vector
        fv = self.fv/10**self.FScale if self._Nseries==1 else self.ff/10**self.FScale

        pntstr = ['']*len(X)
        for k in range(len(X)):
            # Negative frequencies are tricky when _Nseries==1
            #X[k] = abs(X[k]) if self._Nseries==1 else 

            # If user chooses IsFreq option, we convert frequencies back to indices
            if IsFreq:    
                _,X[k] = arrmin(abs( fv - X[k] ))
                _,Y[k] = arrmin(abs( fv - Y[k] ))

            # Create strings to make cute graph labels
            ###pntstr[k] = r'$(%.2f,%.2f)\,\mathrm{%sHz}$' % ( np.sign(X[k])*fv[ abs(X[k]) ], np.sign(Y[k])*fv[ abs(Y[k]) ], ScaleToString(self.FScale) )
            pntstr[k] = r'$(%.2f,%.2f)\,\mathrm{%sHz}$' % ( fv[ X[k] ], fv[ Y[k] ], ScaleToString(self.FScale) )

        if self._Nseries>1:
            X = np.array(X) - len(self.fv)
            Y = np.array(Y) - len(self.fv)

        if CheckNeighbors and len(X)==1:
            X = [X[0],X[0],X[0]-1,X[0]+1]
            Y = [Y[0],Y[0]+1,Y[0],Y[0]]
            self.PlotHelper(whatPlot=whatPlot,X=X,Y=Y,IsFreq=False,CheckNeighbors=CheckNeighbors,fig=fig,ax=ax,
                Ntrials=Ntrials,b2bins=b2bins,cVal=cVal)
            return
        
        if whatPlot=='b2Prob':

            g = np.zeros((Ntrials))

            print('Calculating distribution for {}...      '.format(pntstr[0]))
            for k in range(Ntrials):
                LoadBar(k,Ntrials)
                g[k],_,_ = GetBispec(self.sg,self.BicVec,self.LilGuy,Y[0],X[0],True)
            print('\b\b^]\n')  

            # Limit b^2, create vector, and produce histogram 
            b2lim  = 1 
            b2vec  = np.linspace(0,b2lim,b2bins)
            cnt,_  = np.histogram(g, bins=b2bins, range=(0,b2lim) )

            # Integrate count
            intcnt = sum(cnt) * ( b2vec[1] - b2vec[0] )
            # exp dist -> (1/m)exp(-x/m)
            m = np.mean(g)
            ax.plot(b2vec, cnt/intcnt, linewidth=self.LineWidth, color=self.LineColor[50], marker='x', linestyle='none', label='randomized')
            # More accurate distribution... Just more complicated! (Get to it later...)
            #semilogy(b2vec,(1/m)*exp(-b2vec/m).*(1-b2vec),'linewidth',self.LineWidth,'color','red'); 
            b2dist = (1/m)*np.exp(-b2vec/m)
            ax.plot(b2vec, b2dist, linewidth=self.LineWidth, label=r'$(1/\mu)e^{-b^2/\mu}$')

            N = len(self.tv)
            # Maybe I misunderstand things?
            # chi2dist = np.exp(-b2vec/N) / 2
            # ax.plot(b2vec, chi2dist, linewidth=self.LineWidth, label=r'$(1/2N)\chi^2_2$')
            # This should be chi-squared in general....
            # from scipy.special import gamma
            # X2 = lambda x,nu: x**(nu/2-1) * np.exp(-x/2) / ( 2**(nu/2) * gamma(nu/2) )
            print('nu = ',2 * m**2/np.var(g))

            b2crit = -m*np.log(1 - cVal)
            b2true,_,_ = GetBispec(self.sg,self.BicVec,self.LilGuy,Y[0],X[0],False)
            
            b2noisebase = 10 if self.SpecType=='stft' else self.SampRate / min( abs(self.fv[X[0]]), abs(self.fv[Y[0]]), abs(self.fv[X[0]+Y[0]]) ) # vanMilligen PoP (1995)
            ###b2noisebase = 4*b2true / (1-b2true)**3 ##<- This is from ElgarSebert (1989), 10 is Fackrell (1995), ElgarGuza (1988)
            b2noise = ( b2noisebase / (2*N) )**0.5
            errb2 = N**(-0.5)
            # b2noise = b2noisebase / (2*N)
            # errb2 = 1/N
            ax.axvspan(b2true-errb2, b2true+errb2, facecolor='C2', alpha=0.2)

            # Calculated standard deviation
            ax.axvspan(b2true-(np.var(g))**0.5, b2true+(np.var(g))**0.5, facecolor='C2', alpha=0.2)

            yrange = [0, b2dist[0]]
            #yrange = [1e-3, b2dist[0]*10]
            ax.axvline(b2crit, linewidth=self.LineWidth, color='C1', label=r'$99.9\%$ CI')
            ax.axvline(b2true, linewidth=self.LineWidth, color='C2', label='Measured',)
            ax.axvline(b2noise, linewidth=self.LineWidth, color='C3', label='Noise floor')

            PlotLabels(fig,ax,['$b^2$' + pntstr[0],r'$\mathrm{PDF}$'], self.FontSize, self.CbarNorth)

            ax.set_xlim(0,1)
            ax.set_ylim(yrange[0],yrange[1])

        elif whatPlot=='BvsTime':

            dumt = self.tv/10**self.TScale
            for k in range(len(X)):

                # Calculate "point-out"
                b2est,_,Bi = GetBispec(self.sg,self.BicVec,self.LilGuy,Y[k],X[k],False)
                if Bi is None:
                    print('No bispectral data?')
                    #return

                if self.PlotType in ['abs','imag','real']:
                    umm = eval('np.{}(Bi)'.format(self.PlotType))
                    if self.PlotType == 'abs':
                        ax.semilogy(dumt,umm, linewidth=self.LineWidth, label=pntstr[k], color=self.LineColor[(50+40*k) % 256])
                        ###PlotTimeline(dumt,umm,ax=ax,lw=self.LineWidth,cmap='turbo',cbar=False)
                    else:
                        ax.plot(dumt,umm, linewidth=self.LineWidth, label=pntstr[k], color=self.LineColor[(50+40*k) % 256])
                elif self.PlotType == 'angle':
                    Bi_unwrap = np.unwrap(np.angle(Bi))

                    #ax.plot(dumt,np.unwrap(np.angle(Bi))/np.pi, linewidth=self.LineWidth, linestyle='-.', marker='x', markersize=1, label=pntstr[k], color=self.LineColor[(50+40*k) % 256])
                    ax.plot(dumt,Bi_unwrap/np.pi, linewidth=self.LineWidth, label=pntstr[k], color=self.LineColor[(50+40*k) % 256])

                    N = len(Bi)
                    # Uncertainty as given in Fackrell
                    beta_hi = Bi_unwrap + np.sqrt( (1/b2est - 1) / N )
                    beta_lo = Bi_unwrap - np.sqrt( (1/b2est - 1) / N )
                    ax.fill_between(dumt,beta_hi/np.pi, beta_lo/np.pi, color=self.LineColor[(50+40*k) % 256], alpha=0.2)

            ax.set_xlim([dumt[0],dumt[-1]])


            _,ystr = self.WhichPlot(local=self.bs)
            if self.PlotType == 'angle':
                ystr = ystr + r'/$\pi$'
            tstr = r'$t\, [\mathrm{%ss}]$' % ( ScaleToString(self.TScale) ) if not NoXLabel else ''
            PlotLabels(fig,ax,[tstr,ystr],self.FontSize,self.CbarNorth)

        elif whatPlot=='Phasor':

            _,_,Bi = GetBispec(self.sg,self.BicVec,self.LilGuy,Y[0],X[0],False)
            Bi /= np.max(abs(Bi))

            t = self.tv/10**self.TScale
            tstr = r'$t\, [\mathrm{%ss}]$' % ScaleToString(self.TScale)
            PlotTimeline(np.real(Bi),np.imag(Bi),t=t,fig=fig,ax=ax,lw=self.LineWidth,cmap='twilight',cbar=tstr)
            ##PlotTimeline(np.real(Bi),np.imag(Bi),t=t,fig=fig,ax=ax,lw=self.LineWidth,cmap=self.CMap,cbar=tstr)
            ###ax.plot(np.real(Bi),np.imag(Bi),'o',label=pntstr)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)

            reStr = r'$\mathcal{R}e(\widetilde{\mathcal{B}})/|\widetilde{\mathcal{B}}|_\mathrm{max}$'
            imStr = r'$\mathcal{I}m(\widetilde{\mathcal{B}})/|\widetilde{\mathcal{B}}|_\mathrm{max}$'
            PlotLabels(fig,ax,[reStr,imStr],self.FontSize,self.CbarNorth)

        plt.grid(True)
        if doShow:
            plt.tight_layout()
            if not whatPlot=='Phasor':
                plt.legend()
            if SaveAs is None:
                plt.show()
            else:
                fig.savefig(SaveAs,dpi=self.PlotDPI,bbox_inches='tight')
                plt.close(fig)
        return


    def PlotPointOut(self,X,Y,IsFreq=False,PlotAll=False,SaveAs=None,CheckNeighbors=False):
    # ------------------
    # Plot value of b^2 over time
    # ------------------

        if PlotAll: 

            old_plotType = self.PlotType

            fig = plt.figure(dpi=self.PlotDPI)

            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)

            # print('input is...',X,Y)
            self.PlotHelper('b2Prob',fig=fig,ax=ax1,X=X,Y=Y,IsFreq=IsFreq,CheckNeighbors=CheckNeighbors)

            # For some reason this changes X and Y???
            # print('but now is...',X,Y)

            self.PlotHelper('Phasor',fig=fig,ax=ax3,X=X,Y=Y,CheckNeighbors=CheckNeighbors)

            self.PlotType = 'angle'
            self.PlotHelper('BvsTime',fig=fig,ax=ax2,X=X,Y=Y,CheckNeighbors=CheckNeighbors)

            self.PlotType = 'abs'
            self.PlotHelper('BvsTime',fig=fig,ax=ax4,X=X,Y=Y,CheckNeighbors=CheckNeighbors)

            self.PlotType = old_plotType

            ax1.legend(fontsize=2*self.FontSize//3)
            #ax2.legend(fontsize=2*self.FontSize//3)
            #ax3.legend()
            ax4.legend(fontsize=2*self.FontSize//3,loc='upper center',bbox_to_anchor=(0.35, 1.5),ncol=2)

            # figSpace = 0.1
            # fig.subplots_adjust(hspace=figSpace,wspace=figSpace)

            plt.tight_layout()
            if SaveAs is None:
                plt.show()
            else:
                fig.savefig(SaveAs,dpi=self.PlotDPI,bbox_inches='tight')
                plt.close(fig)

        elif self.PlotType == 'hybrid':
            self.PlotHelper('Phasor',X=X,Y=Y,IsFreq=IsFreq,SaveAs=SaveAs)

        elif self.PlotType == 'bicoh':
            self.PlotHelper('b2Prob',X=X,Y=Y,IsFreq=IsFreq,SaveAs=SaveAs)

        elif self.PlotType in ['abs','real','imag','angle']:
            self.PlotHelper('BvsTime',X=X,Y=Y,IsFreq=IsFreq,SaveAs=SaveAs,CheckNeighbors=CheckNeighbors)

        return X,Y
        


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
        
        self.PlotBispec(fig,self.AxHands[0],normb2=self.NormBic)

        self.PlotSpectro(fig,self.AxHands[1],vLim=self.SpecVLim)
        if self.PlotSlice is not None:
            It = self.PlotSlice

            t  = self.tv/10**self.TScale
            f  = self.fv/10**self.FScale
            dt = (self.SubInt/self.SampRate)/10**self.TScale

            self.AxHands[1].axvline(t[It], color='white', linewidth=1.5)
            self.AxHands[1].axvline(t[It]+dt, color='white', linewidth=1.5)

        self.PlotPowerSpec(fig,self.AxHands[2])       

        plt.tight_layout()
        plt.show()
        self.NewGUICax = False
        return


    def PlotGUI(self):
    # ------------------
    # GUI test
    # ------------------
        fig = plt.figure(dpi=self.PlotDPI) ###,figsize=[9,6])

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(224)

        # ax1 = plt.subplot(211)
        # ax2 = plt.subplot(223)
        # ax3 = plt.subplot(224)

        # Save figure and axes with object
        self.Figure = fig
        self.AxHands = [ax1, ax2, ax3]
        self.CaxHands = [None,None]

        cid = fig.canvas.mpl_connect('button_press_event', self.ClickPlot)
        pid = fig.canvas.mpl_connect('key_press_event', self.SwitchPlot)
        
        self.NewGUICax = True
        self.RefreshGUI()
        return


    def FindMaxInRange(self,FxLo,FxHi,FyLo,FyHi,useb2=True):
    # ------------------
    # Finds maximum bicoherence in given range
    # Good for tracking a particular feature
    # ------------------
        
        # TODO: Set up for cross-b^2 and tricoherence

        # Transform to desired scaling
        dum = self.fv / 10**self.FScale
        vx = (dum>FxLo) * (dum<FxHi) * 1
        vy = (dum>FyLo) * (dum<FyHi) * 1
        
        vy = vy[0:len(vy)//2]
        
        # bicMask = vy.' * vx;
        # dumBic = bicMask .* bic.bc
        # [maxVal,Ic] = max(max(dumBic))
        # [~,Ir] = max(max(dumBic.'))
        bicMask = np.outer(vy,vx)
        dumBic = bicMask * self.bc if useb2 else bicMask * abs(self.bs)

        # Two ways to do this!
        maxVal = dumBic.max()
        I = dumBic.argmax()
        # Ir = I // dumBic.shape[1]
        # Ic = I %  dumBic.shape[1]
        Ir,Ic = np.unravel_index(I, dumBic.shape)
        
        fX = dum[Ic]
        fY = dum[Ir]
        
        print('Maximum found! Value is %.4f @ row = %d, column = %d' % (maxVal,Ir,Ic))
        print('Scaled frequency equivalents are freqX = %.2f, freqY = %.2f' % (fX,fY))

        return Ir,Ic,maxVal
        
        
        # function [Ir,Ic,maxVal] = FindMaxAboveLim(bic,FreqLow)
        # % ------------------
        # % Finds maximum bicoherence above frequency limit
        # % Good for ignoring low-frequency shizz
        # % ------------------
        
        # % Get maxima in each column
        # [m,I] = max(bic.bc);
        # % Sort in descending order
        # [mSorted,IX] = sort(m,'descend');
        
        # stillLooking = true;
        # cnt = 1;
        # % Walk down list of maxima, choose the first above limit
        # while stillLooking
        #     % Get column of current maximum
        #     testCol = IX(cnt);
        #     if ( bic.fv(testCol) / 10^bic.FScale ) > FreqLow
        #         % Boom!
        #         stillLooking = false;
        #     else
        #         cnt = cnt + 1;
        #     end          
        # end
        
        # maxVal = mSorted(cnt);
        # Ic = testCol;
        # Ir = I(testCol);
        
        # fX = bic.fv(Ic) / 10^bic.FScale;
        # fY = bic.fv(Ir) / 10^bic.FScale;
        
        # fprintf('Maximum found! Value is %.4f @ row = %d, column = %d\n',maxVal,Ir,Ic);
        # fprintf('Scaled frequency equivalents are freqX = %.2f, freqY = %.2f\n',fX,fY);
        
        # end % FindMaxAboveLim


    def SizeWarnPrompt(self,n):
    # ------------------
    # Prompt for CPU health
    # ------------------
        qwer = messagebox.askokcancel('Question','FFT elements exceed {}! ({}) Continue?'.format(self._WarnSize,n))
        if not qwer:
            print('Operation terminated by user.')
            self._RunBicAn = False
        #qwer.withdraw()
        #qwer.destroy()
        return         # Bail if that seems scary! 


    def ClickPlot(self,event):
    # ------------------
    # Callback for clicks
    # ------------------
        ax = event.inaxes
        print('ax is',ax)
        if ax == self.AxHands[0]: # Check bispectrum
            fx = event.xdata
            fy = event.ydata
            buf = 'fx = %.3f, fy = %.3f' % (fx, fy)

            f = self.fv/10**self.FScale
            if self._Nseries>1:
                f = self.ff/10**self.FScale
                # Need to subtract something from index now!!!!

            _,Ix = arrmin( abs(f-fx) )
            _,Iy = arrmin( abs(f-fy) )

            # Debug/diagnostic data
            print(buf)
            print(Ix,Iy)
            print('button=',event.button)

            ax.plot(f[Ix],f[Iy],'o',linestyle='none',color='white',markerfacecolor='none')

            ###self.AxHands[1].plot(f[Ix],f[Iy],'-.',color='red')

            if self._Nseries==1:
                # Plot lines on spectrogram and PSD
                for k in [Ix,Iy,Ix+Iy]:
                    self.AxHands[1].axhline(f[k], color='red', linewidth=1.5)
                    self.AxHands[2].axvline(f[k], color='red', linewidth=1.5)

            # 1 = MouseButton.LEFT
            # 3 = MouseButton.RIGHT
            if event.button==1:
                self.PlotPointOut([Ix],[Iy])
            elif event.button==3:
                self.PlotPointOut([Ix],[Iy],PlotAll=True)

        elif ax == self.AxHands[1]: # Check spectrogram
            tx = event.xdata
            t  = self.tv/10**self.TScale
            _,self.PlotSlice = arrmin( abs(t-tx) ) # Find closest point in time
            self.RefreshGUI()

        else:
            print('No callback there!')
        return    


    def SwitchPlot(self,event):
    # ------------------
    # Callback for keypress
    # ------------------
        key  = event.key
        opts = 'BARIPMSH'
        sel  = '!@#'
        if key in opts:
            ind = opts.index(key)

            figs = ['bicoh','abs','real','imag','angle','mean','std','hybrid']
            self.PlotType = figs[ind]
        elif key in sel:
            ind = sel.index(key)
            if ind<self._Nseries:
                self.PlotSig = ind
            else:
                print('Not available!')
        elif key == 'h':
            choiceBox()
            print('Some kind of help menu here!')
        elif key == 'X': # Reset GUI
            self.PlotSlice = None
            self.BicOfTime = False
            self.PlotType = 'bicoh'
        elif key == 'T':
            self.BicOfTime = not self.BicOfTime
        elif key == 'right':
            self.PlotSlice = 0 if self.PlotSlice is None else (self.PlotSlice + 10) % len(self.tv) 
        elif key == 'left':
            self.PlotSlice = len(self.tv)-1 if self.PlotSlice is None else (self.PlotSlice - 10) % len(self.tv)
        else:
            return

        # Activate!
        self.RefreshGUI()
        return




    def choiceBox(self):

        root = tk.Tk()

        v = tk.IntVar()
        v.set(1)  # initializing the choice, i.e. Python

        languages = [('viridis', 0),
                     ('gnuplot2', 1),
                     ('PiYG', 2),
                     ("C++", 104),
                     ("C", 105)]

        tk.Label(root, 
                 text="""Choose your favourite 
        programming language:""",
                 justify = tk.LEFT,
                 padx = 20).pack()

        for language, val in languages:
            tk.Radiobutton(root, 
                           text=language,
                           padx = 20, 
                           variable=v, 
                           command=self.DumFunc,
                           value=val).pack(anchor=tk.W)


        #root.mainloop()
        return

    def DumFunc(self):
        #self.CMap = 
        print(event)
        return


    def CheckCouple(self,f):
    # ------------------
    # For a given test vector of freqs, check nth order coupling
    # ------------------
        f = f if self._Nseries==1 else np.array(X) - len(self.fv)

        n = len(f)
        mask = bin_mat(n)
        L = 2**(n-1)
        out = np.zeros(L)
        for k in range(L):
            dum = f * mask[k,:]
            out[k],_,_ = GetPolySpec(self.sg,dum,self.LilGuy)
            print(dum,'~>',out[k])
        print('Mean is ',np.mean(out))
        return out


# Module methods

def FileDialog():
# ------------------
# Ganked from StackExchange...
# ------------------
    root = tk.Tk()
    root.withdraw()
    # Build a list of tuples for each file type the file dialog should display
    my_ftypes = [('all files', '.*'), ('text files', '.txt'), ('dat files', '.dat')]
    # Ask the user to select a single file name
    ans = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select a file:",filetypes=my_ftypes)  
    return ans


def PlotLabels(fig,ax,strings=['x','y'],fsize=20,cbarNorth=False,im=None,cax=None,fweight='normal',tickweight='bold',cbarweight='none',grid=True):
# ------------------
# Convenience function
# ------------------
    n = len(strings)

    # Reduce fontsize for 3D plots
    fsize = fsize if n<4 else 3*fsize//4

    # Kill grid if n>2
    grid = grid if n<=2 else False

    # Initialize list for tick label info
    labels = []

    ax.set_xlabel(strings[0], fontsize=fsize, fontweight=fweight)
    if n>1:
        ax.set_ylabel(strings[1], fontsize=fsize, fontweight=fweight)
    if n==3:
        if cax is None:
            divider = make_axes_locatable(ax)
            cbarloc = 'top' if cbarNorth else 'right'
            cax = divider.append_axes(cbarloc, size='5%', pad=0.05)
        else:
            cax.clear()
        if cbarNorth:
            fig.colorbar(im, cax=cax, orientation='horizontal')   
            cax.xaxis.set_label_position('top') 
            cax.xaxis.tick_top()     
            cax.set_xlabel(strings[2], fontsize=fsize, fontweight=fweight)
        else:
            fig.colorbar(im, cax=cax)
            cax.set_ylabel(strings[2], fontsize=fsize, fontweight=fweight)
        cax.tick_params(labelsize=3*fsize//4)
        if cbarweight=='ticks':
            labels += cax.get_xticklabels()
            labels += cax.get_yticklabels()
    if n==4: # Must be trispectrum

        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(strings[2], fontsize=fsize, fontweight=fweight, rotation=90)
        cbar = fig.colorbar(im,cax=None,ax=ax,shrink=0.75,label=strings[3])

        cbar.ax.set_ylabel(strings[3], fontsize=fsize, fontweight=fweight)
        cbar.ax.tick_params(labelsize=3*fsize//4)

        ###labels += ax.get_zticklabels()
        labels += cbar.ax.get_xticklabels()
        labels += cbar.ax.get_yticklabels()

    ax.tick_params(labelsize=9*fsize//10)
    ax.minorticks_on()

    # Append ticklabels
    labels += ax.get_xticklabels()
    labels += ax.get_yticklabels()    

    # THE MAGIC HAPPENS HERE!
    # This is a robust solution, and doesn't have bad practice "set current axis" calls
    for label in labels:
        label.set_fontweight(tickweight)

    if grid:
        ax.grid(visible=True,which='minor',linewidth=0.5,color=[0.9,0.9,0.9])
        ax.grid(visible=True,which='major')
    return cax


def SignalGen(fS=1,tend=100,Ax=1,fx=1,Afx=0,Ay=0,fy=0,Afy=0,Az=0,Ff=0,noisy=2):
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

    # New addition for quadratic coupling tests
    #phi = np.pi/4
    #z = Az*np.abs(np.sin(2*np.pi*Ff*t))**4 * x*y + 0*np.sin(2*np.pi*(fx*t + fy*t + phi))

    sig = x + y + z + noisy*(0.5*np.random.random(len(t)) - 1)
    #sig = np.reshape(sig, ( len(sig), 1 )) # Output Nx1 numpy array
    return sig,t,float(fS)


def TestSignal(whatsig):
# ------------------
# Provides FM test signal
# ------------------
    fS   = 200
    tend = 100
    noisy = 2
    f1 = 19
    f2 = 45
    dum = whatsig.lower()
    if dum == 'classic':
        inData,t,_ = SignalGen(fS,tend,fx=f2,Afx=6,Ay=1,fy=f1,Afy=10,Az=1,Ff=1/20)
    elif dum == 'tone':
        inData,t,_ = SignalGen(fS,tend,fx=f1)
    elif dum == 'noisy':
        inData,t,_ = SignalGen(fS,tend,fx=f1,noisy=5*noisy)
    elif dum == '2tone':
        inData,t,_ = SignalGen(fS,tend,fx=f1,Ay=1,fy=f2)
    elif dum in ['3tone','3tone_short']:
        tend = 5 if dum=='3tone_short' else tend
        inData,t,_ = SignalGen(fS,tend,fx=f1,Ay=1,fy=f2,Az=1)
    elif dum == '4tone':
        # old was 13,17,54
        # then was 15,25,45
        x1,t,_ = SignalGen(fS,tend,fx=12,noisy=0)
        x2,_,_ = SignalGen(fS,tend,fx=32,noisy=0)
        x3,_,_ = SignalGen(fS,tend,fx=40,noisy=0)
        x4,_,_ = SignalGen(fS,tend,fx=12+32+40,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x1 + x2 + x3 + x4 + nz

    elif dum == 'ntone':
        # ask how many

        # get n freqs

        # 
        print('Not an option yet!!!')

    elif dum == 'line':
        inData,t,_ = SignalGen(fS,tend,fx=f1,Ay=1,fy=f2,Afy=10,Az=1,Ff=1/20)
    elif dum in ['circle','circle_oversample']:
        fS = 10*fS if dum=='circle_oversample' else fS
        inData,t,_ = SignalGen(fS,tend,fx=f1,Afx=10,Ay=1,fy=f2,Afy=10,Az=1,Ff=1/20)
    elif dum == 'fast_circle':
        tend = 20
        inData,t,_ = SignalGen(fS,tend,fx=f1,Afx=5,Ay=1,fy=f2,Afy=5,Az=1,Ff=5/20)
    elif dum == 'quad_couple':
        x,t,_ = SignalGen(fS,tend,fx=f1,noisy=0)
        y,_,_ = SignalGen(fS,tend,fx=f2,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + x * y + nz
    elif dum == 'd3dtest':
        fS = 500
        f1 = 97
        f2 = 84
        Ff = 1/20
        Az = -1
        x,t,_ = SignalGen(fS,tend,fx=f1,noisy=0)
        y,_,_ = SignalGen(fS,tend,fx=f2,noisy=0)
        z,_,_ = SignalGen(fS,tend,Ax=0,Ay=1,fy=f1-f2,noisy=0)
        A,_,_ = SignalGen(fS,tend,fx=Ff,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + Az * (A**4)*x*y + 0.0*z + nz/2
    elif dum == 'amtest':
        fS = 500
        f1 = 15
        f2 = 93
        f3 = 108
        x,t,_ = SignalGen(fS,tend,fx=f1,noisy=0)
        y,_,_ = SignalGen(fS,tend,fx=f2,noisy=0)
        z,_,_ = SignalGen(fS,tend,fx=f3,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + x*y + nz
    elif dum == 'cube_couple':
        x,t,_ = SignalGen(fS,tend,fx=13,noisy=0)
        y,_,_ = SignalGen(fS,tend,fx=17,noisy=0)
        z,_,_ = SignalGen(fS,tend,fx=54,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + z + x*y*z + nz
    elif dum == 'coherence':
        x,t,_ = SignalGen(fS,tend,fx=f1)
        y,_,_ = SignalGen(fS,tend,fx=f2)
        z,_,_ = SignalGen(fS,tend,fx=f1)
        inData = np.zeros( (len(t), 2) )
        inData[:,0] = x
        inData[:,1] = y + z
    elif dum == 'cross_2tone':
        x,t,_ = SignalGen(fS,tend,fx=f1)
        y,_,_ = SignalGen(fS,tend,fx=f2)
        inData = np.zeros( (len(t), 2) )
        inData[:,0] = x
        inData[:,1] = x + y
    elif dum in ['cross_3tone','cross_3tone_short']:
        tend = tend if dum=='cross_3tone' else 5
        x,t,_ = SignalGen(fS,tend,fx=f1)
        y,_,_ = SignalGen(fS,tend,fx=f2)
        z,_,_  = SignalGen(fS,tend,1,fx=f1+f2)
        inData = np.zeros( (len(t), 3) )
        inData[:,0] = x 
        inData[:,1] = y 
        inData[:,2] = z
    elif dum == 'quad_couple_circle':
        x,t,_  = SignalGen(fS,tend,fx=f1,Afx=10,Ff=1/20)
        y,_,_  = SignalGen(fS,tend,Ax=0,fx=0,Ay=1,fy=f2,Afy=10,Ff=1/20)
        z,_,_  = SignalGen(fS,tend,Ax=0,fx=f1,Afx=10,Ay=0,fy=f2,Afy=10,Az=1,Ff=1/20)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + x * y + nz
    elif dum == 'quad_couple_circle2':
        x,t,_  = SignalGen(fS,tend,fx=f1,Afx=10,Ff=1/20)
        y,_,_  = SignalGen(fS,tend,Ax=0,fx=0,Ay=1,fy=f2,Afy=10,Ff=1/20)
        z,_,_  = SignalGen(fS,tend,Ax=0,fx=f1,Afx=10,Ay=0,fy=f2,Afy=10,Az=1,Ff=1/20)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + x * y + nz
    elif dum == 'cross_circle':
        x,t,_  = SignalGen(fS,tend,fx=f1,Afx=10,Ff=1/20)
        y,_,_  = SignalGen(fS,tend,Ax=0,fx=0,Ay=1,fy=f2,Afy=10,Ff=1/20)
        z,_,_  = SignalGen(fS,tend,Ax=0,fx=f1,Afx=10,Ay=0,fy=f2,Afy=10,Az=1,Ff=1/20)
        inData = np.zeros( (len(t), 3) )
        inData[:,0] = x
        inData[:,1] = y
        inData[:,2] = z
    elif dum == 'helix':
        x,t,_  = SignalGen(fS,tend,fx=f1,Afx=10,Ff=1/20,noisy=0)
        y,_,_  = SignalGen(fS,tend,Ax=0,fx=0,Ay=1,fy=f2,Afy=10,Ff=1/20,noisy=0)
        z,_,_  = SignalGen(fS,tend,Ax=0,fx=f1,Afx=10,Ay=0,fy=f2,Afy=10,Az=1,Ff=1/20,noisy=0)
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = x + y + z + x*y*z + nz
    else:
        print('***WARNING*** :: "{}" test signal unknown... Using single tone..'.format(whatsig)) 
        inData,t,fS = SignalGen(fS,tend,1,22,0,0,0,0,0,0,0)
    return inData,t,float(fS)


def ApplySTFT(sig,samprate=1,subint=512,step=256,nfreq=256,t0=0,detrend=False,errlim=1e15):
# ------------------
# STFT static method
# ------------------
    N = min(sig.shape)
    M = 1 + (max(sig.shape) - subint)//step
    lim  = nfreq                    # Most likey, lim = |_ Nyquist/res _|
    time_vec = np.zeros(M)          # Time vector
    err  = np.zeros((M,N))          # Mean information
    spec = np.zeros((lim,M,N),dtype=complex)      # Spectrogram
    fft_coeffs = np.zeros((lim,N),dtype=complex)  # Coeffs for slice
    afft = np.zeros((lim,N))        # Coeffs for slice
    Ntoss = 0                       # Number of removed slices
    
    win = HannWindow(subint)        # Apply Hann window
    
    print('Applying STFT...      ')
    for m in range(M):
        LoadBar(m,M)

        time_vec[m] = t0 + m*step/samprate
        for k in range(N):
            Ym = sig[m*step : m*step + subint, k] # Select subinterval    
            # Removing this b/c it negates the point of nfreq
            #Ym = Ym[0:nfreq]            # Take only what is needed for res
            if detrend:                  # Remove linear least-squares fit
                Ym = ApplyDetrend(Ym)
            mean = sum(Ym)/len(Ym)
            Ym = win*(Ym-mean)   # Remove DC offset, multiply by window

            DFT = np.fft.fft(Ym) # Compute FFT
            DFT /= len(DFT)      # Normalize by vector length

            fft_coeffs[0:lim,k] = DFT[0:lim]     # Get interested parties
            dumft    = abs(fft_coeffs[:,k])**2   # Dummy for abs(coeffs)^2
            err[m,k] = sum(dumft)/len(dumft)     # Mean of PSD slice

            if err[m,k]>=errlim:
                fft_coeffs[:,k] = 0*fft_coeffs[:,k] # Blank if mean excessive
                Ntoss += 1

            afft[:,k]  += dumft                  # Welch's PSD
            spec[:,m,k] = fft_coeffs[:,k]        # Build spectrogram

    print('\b\b\b^]\n')
    
    freq_vec = np.arange(lim)*samprate/subint
    #freq_vec = freq_vec[0:lim] 
    afft /= M     
    return spec,afft,freq_vec,time_vec,err,Ntoss


def ApplyCWT(sig,samprate=1,sigma=3.14,limFreq=2,alphaExp=0.5):
# ------------------
# Wavelet static method
# ------------------
    Nsig,N = sig.shape
    nyq    = Nsig//2

    lim = int(Nsig/limFreq)

    f0 = samprate/Nsig
    freq_vec = f0 * np.arange(nyq) # Frequency vector as calculated by FFT
    
    acwt = np.zeros((lim,N))
    CWT  = np.zeros((lim,nyq,N),dtype=complex)

    # Morlet wavelet in frequency space
    ###Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma) * a**(alphaExp-0.5) * np.exp( -2 * np.pi**2 * sigma**2 * ( freq_vec/a - f0)**2 )
    Psi = lambda a: (np.pi**0.25)*np.sqrt(2) * a**(alphaExp-0.5) * np.exp( -2 * np.pi**2 * (sigma/a)**2 * ( np.arange(nyq) - a)**2 )

    for k in range(N):
        fft_sig = np.fft.fft(sig[:,k])
        fft_sig = fft_sig[0:nyq]

        # Deal with DC bin?
        # CWT[0,:,k] = (np.pi**0.25)*np.sqrt(2*sigma) * fft_sig[0]

        print('Applying CWT...      ')
        for a in range(lim-1):
            LoadBar(a,lim)
            # Apply for each scale (read: frequency)
            dum = np.fft.ifft(fft_sig * Psi(a+1)) ###/ Nsig         # Linear scale (f_a = a*f0)
            #dum = np.fft.ifft(fft_sig * Psi( 2**((a+1)/12) ))   # Equal-tempered
            #dum = np.fft.ifft(fft_sig * Psi( (a+1)/10 ) )
            CWT[a+1,:,k] = dum

            acwt[a+1,k]  = sum(abs(dum)**2) / len(dum)
        print('\b\b\b^]\n')

    time_vec = 2 * np.arange(nyq)/samprate
    return CWT,acwt,freq_vec[0:lim],time_vec


def SpecToCoherence(spec,lilguy=1e-6):
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


def SpecToBispec(spec,v=[0,0,0],lilguy=1e-6):
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

            Bi  = p1 * p2 * np.conj(s)
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


def SpecToCrossBispec(spec,v=[0,0,0],lilguy=1e-6):
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

                Bi  = p1 * p2 * np.conj(s)
                e12 = abs( p1 * p2 )**2   
                e3  = abs(s)**2  

                Bjk = sum(Bi)                    
                E12 = sum(e12)             
                E3  = sum(e3)                     

                b2[j+nfreq-1,k+nfreq-1] = (abs(Bjk)**2)/(E12*E3+lilguy)

                B[j+nfreq-1,k+nfreq-1] = Bjk

    B = B/slices
    print('\b\b\b^]\n')                    
    return b2,B


def SpecToTrispec(spec,v=[0,0,0],lilguy=1e-6):
# ------------------
# Turns spectrogram to t^2
# ------------------
    nfreq,slices,_ = spec.shape

    lim = nfreq

    T  = np.zeros((lim//2,lim,lim//3), dtype=complex)
    t2 = np.zeros((lim//2,lim,lim//3))
    
    print('Calculating tricoherence...      ')     
    for j in range(lim//2):
        LoadBar(j,lim//2);
        
        for k in np.arange(j,lim-j):

            for n in range(lim//3):

                if j+k+n<lim and n<=j and n<=k:
            
                    p1 = spec[k,:,v[0]]
                    p2 = spec[j,:,v[1]]
                    p3 = spec[n,:,v[2]]
                    s  = spec[j+k+n,:,v[3]]

                    # See Kravtchenko-Berejnoi et al. [1995]
                    # Ti   = (p1) * (p2) * np.conj(p3) * conj(s);
                    # e123 = abs((p1) * (p2) * conj(p3))**2;
                    # e4   = abs(s)**2;  

                    Ti   = p1 * p2 * p3 * np.conj(s)
                    e123 = abs(p1 * p2 * p3)**2
                    e4   = abs(s)**2

                    Tjkn = sum(Ti)                    
                    E123 = sum(e123)             
                    E4   = sum(e4)                     

                    t2[j,k,n] = ( abs(Tjkn)**2 ) / ( E123*E4 + lilguy ) 
                    T[j,k,n]  = Tjkn

    T = T/slices
    print('\b\b\b^]\n')     

    return t2,T              


def GetBispec(spec,v=[0,0,0],lilguy=1e-6,j=0,k=0,rando=False):
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
        p1 = abs(p1)*np.exp( 2j*np.pi* (2*np.random.random( p1.shape ) - 1) )
        p2 = abs(p2)*np.exp( 2j*np.pi* (2*np.random.random( p2.shape ) - 1) )
        s  = abs(s)* np.exp( 2j*np.pi* (2*np.random.random( s.shape  ) - 1) )

    Bi  = p1*p2*np.conj(s)
    e12 = abs(p1*p2)**2
    e3  = abs(s)**2

    B   = sum(Bi)                 
    E12 = sum(e12)            
    E3  = sum(e3)                      

    w = (abs(B)**2)/(E12*E3+lilguy)
    
    B = B/len(Bi)
    return w,B,Bi 


def GetPolySpec(spec,f,lilguy=1e-6,rando=False,v=[]):
# ------------------
# Calculates the nth-order coherence of a given (f1,f2,...,fn) value
# ------------------

    N = len(f)
    sumFreq = sum(f)

    if len(v)==0 or not len(v)==N+1:
        # Default to all first time series if issue
        v = (N+1) * [0]

    getCoeff = lambda i,n: np.real( spec[abs( i ),:,v[n]] ) + 1j*np.sign( i )*np.imag( spec[abs( i ),:,v[n]] )

    s = np.conj( getCoeff(sumFreq,N) )
    if rando:
        s  = abs(s) * np.exp( 2j*np.pi* (2*np.random.random( s.shape  ) - 1) )

    nSpec_i = np.ones( s.shape , dtype=complex)

    for k in range(N):
        p = getCoeff( f[k], k )

        if rando:
            p = abs(p)*np.exp( 2j*np.pi* (2*np.random.random( p.shape ) - 1) )

        nSpec_i  *= p

    e1 = abs( nSpec_i )**2
    e2 = abs( s )**2

    nSpec  = sum( nSpec_i * s )                 
    E1 = sum( e1 )            
    E2 = sum( e2 )                      

    nCoh = ( abs(nSpec)**2 ) / ( E1*E2 + lilguy )
    
    nSpec /= len( nSpec_i )
    return nCoh,nSpec,nSpec_i*s


def HannWindow(N,q=2):
# ------------------
# Hann window
# ------------------
    win = (np.sin(np.pi*np.arange(N)/(N-1)))**q
    return win


def ApplySimpleFilter(x,fS=1.0,f0=0.25,fband=0.1):
# ------------------
# Applies quick & dirty bandpass with brickwall
# ------------------
    N = len(x)
    fftx = np.fft.fft(x)

    plt.plot(abs(fftx))

    Klo  = np.ceil( (N/fS) * (f0 - fband ) ).astype(int)
    Khi  = np.ceil( (N/fS) * (f0 + fband ) ).astype(int)

    # fftx[0:Klo] = 0
    # fftx[(-Klo):-1] = 0
    # fftx[Khi:(-Khi)] = 0
    fftx[0:Klo] = 0
    fftx[(-1-Klo+1):-1] = 0
    fftx[Khi:(-1-Khi+1)] = 0

    plt.plot(abs(fftx))
    plt.show()

    return np.real(np.fft.ifft(fftx))    # Invert it!


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
    tags = ['f',[],[],'p',[],[],'n',[],[],'$\mu$',[],[],'m','c','d','', [],'h','k',[],[],'M',[],[],'G',[],[],'T',[],[],'P',[],[],'E']
    s = tags[15+scale]
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


def PlotTimeline(x,y,t=None,fig=None,ax=None,lw=2,cmap='turbo',cbar=None,fsize=14):
# ------------------
# Plot line with color
# ------------------
    ### Sourced from https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
    from matplotlib.collections import LineCollection

    if fig is None:
        fig, ax = plt.subplots()

    if t is None:
        t = np.linspace(0,1,x.shape[0]) # "time" variable

    # set up a list of (x,y) points
    points = np.array([x,y]).transpose().reshape(-1,1,2)

    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1)

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap))
    lc.set_array(t) # color the segments by our parameter
    lc.set_linewidth(lw)

    # plot the collection
    line = ax.add_collection(lc) # add the collection to the plot
    if cbar is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.05)
        fig.colorbar(line, cax=cax, orientation='horizontal')   
        cax.xaxis.set_label_position('top') 
        cax.xaxis.tick_top()     
        cax.set_xlabel(cbar,fontsize=fsize) ###, fontweight=fweight)
        # fig.colorbar(line, ax=ax, label=cbar)


def RunDemo():
# ------------------
# Demonstration
# ------------------
    b = BicAn('circle')
    return b


def arrmin(arr):
# ------------------
# Matlab-esque min()
# ------------------   
    m = min(arr)
    index = arr.tolist().index( m )
    return m,index


def bin_mat(n):
  # Creates matrix of ...
  dum = ((np.arange(2**(n-1),2**n).reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)[:,::-1]
  dum[dum==0] = -1
  return dum


def diff_to_sum_vec(v):
  v[0] = sum(v)
  return v


def nRandSumLessThanUnity(n):
# ------------------
# Outputs n numbers whose sum is < 1
# ------------------
    foundIt = False
    while not foundIt:
        
        dum = np.random.random(n)

        if sum(dum)<=1:
            foundIt = True
            dum.sort()
            dum = dum[::-1]
            return dum


def DrawSimplex(flim):
# ------------------
# Draws simplex for trispectrum
# ------------------
    plt.plot([0,flim/3],     [0,flim/3],     [0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim,flim/3],  [0,flim/3],     [0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,flim/3],[flim/2,flim/3],[0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,0],     [flim/2,0],     [0,0],     color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,flim],  [flim/2,0],     [0,0],     color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([0,flim],       [0,0],          [0,0],     color=[0.5,0.5,0.5], lw=2.5)