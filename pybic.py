# -*- coding: utf-8 -*-
r"""Polyspectral analysis toolkit for Python.

.. code-block:: text

    ______     ______ _      
    | ___ \    | ___ (_)          
    | |_/ /   _| |_/ /_  ___      
    |  __/ | | | ___ \ |/ __|               v2.1 (c) 2022-2026
    | |  | |_| | |_/ / | (__                
    \_|   \__, \____/|_|\___|              G. Riggs & T. Matheny
           __/ |                           
          |___/             

**PyBic** is an open-source module specializing in signal processing, 
with particular emphasis on polyspectral analysis. 

The *Bic* in ``PyBic`` refers to bicoherence analysis, which is by 
far the most common use of the polyspectrum. Explicitly, we use

.. code-block:: text

    The bispectrum

    B_xyz(f1,f2) = < X(f1)Y(f2)Z(f1+f2)* >, 

    where x,y,z are time series with corresponding Fourier transforms 
    X,Y,Z, and <...> denotes averaging in time.

    The (squared) bicoherence spectrum

    b^2_xyz(f1,f2) =           |B_xyz(f1,f2)|^2
                             --------------------
                   ( <|X(f1)Y(f2)|^2> <|Z(f1+f2)|^2> + eps ),

    where eps is a small number meant to prevent 0/0 = NaN catastrophe

For more information and references on the history, theory, utility, 
and implementation of polyspectra, please see our publication in 
*Computer Physics Communications*, `RiggsKoepkeMatheny2026`_.

Also check out our `GitHub repo`_ and `Read the Docs`_ pages!

To run the demo from the shell, use

.. code-block:: shell-session

    $ python3 pybic.py

or, alternatively, in Python

.. code-block:: python

    import pybic as bic
    b = bic.RunDemo()

Additionally, we've developed Jupyter notebooks with a `guided tour`_
of PyBic and a `demonstration`_ of the :func:`~pybic.Plot` function.

.. todo::
    :collapsible: closed

    * Add minimum threshold keyword to PlotBispec() method to mask b^2 below noise floor (none)
    * Add colormap picker in PlotGUI() with SHIFT + c, say (none)
    * Swap out matplotlib widgets for full tkinter GUI =^x (some)
    * Figure out setter functions (some)
    * Configure warnings (none)
    * Implement some kind of check for Raw data! Should eliminate string, etc. (done)
    * Fix colorbar axes overplotting each refresh (done)
    * Fix issue with colorbar labels when calling RefreshGUI() (done)
    * Add buttons and callbacks from Matlab (some)
    * Swap out "dum" variables for more literate ones
    * Comment the code!!! (some)
    * Fix butt-ugly inputs to PlotPointOut children! (done)
    * Flag for base units (maybe not based in time, say)
    * Antialiased option! (none)
    * Bispectrogram (none)
    * Auto-filters! (none)
    * Video output (none)
    * Base units (none)
    * Cross stuff in :func:`BicAn.FindMaxInRange` (none)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

.. _RiggsKoepkeMatheny2026:
    https://doi.org/10.1016/j.cpc.2026.110097

.. _GitHub repo:
    https://github.com/rigzridge/pybic

.. _Read the Docs:
    https://pybic.readthedocs.io

.. _guided tour:
    https://colab.research.google.com/drive/1GnJddGDVVIWK44B-_0Mfoe-tLKWoXFrb?usp=sharing

.. _demonstration:
    https://colab.research.google.com/drive/1NJmjnkhD9wWd_uYRYDWSOEatzS_5Nzm3?usp=sharing

"""

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Version History
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6/04/2026 -> Added a bunch of features to PlotTimeline()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6/03/2026 -> Cleaned up some broken docstrings (""" -> r""" and done!)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6/02/2026 -> Finally debugged colormap radiobuttons; fixed interpolation 
# issue in PlotInstFreq(); reverted 'quad_couple' test signal for Colab;
# better input checking for CheckCouple() [won't accept bins > NFreq];
# removed redundant method ApplyBandpass() -> renamed ApplySimpleFilter() 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/27/2026 -> Fixed "plot3d = True" option in PlotBispec()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/22/2026 -> Removed LoadBarOLD() and added more docstrings 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/01/2026 -> Fixed plot labels in various Plot...() methods to better reflect 
# normalized analysis, ie, f given in units of fs; LoadBar output cleaned up;
# removed vestigial access to FreqRes attribute; SizeWarnPrompt debugged;
# eliminated unused Verbose attribute; testing detection of time vector
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4/27/2026 -> Better docstrings, hopefully we're done this week!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4/22/2026 -> Big changes to github repo, getting everything switched to 
# docstrings for conveniently generated documentation from ReadTheDocs
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2/26/2026 -> Paper in Computer Physics Communications available online!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1/18/2026 -> Accepts more general colormaps, ie, mpl.colors.ListedColormap
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1/15/2026 -> Added phase/amplitude test signals from paper + PlotPhaseDist()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1/14/2026 -> Incorporated bootstrap estimates of b2 and B into PDF plots
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 12/05/2025 -> Fixed labeling for 3D trajectories, ie, ['x','y','z','col']
# produces no colorbar when 'col' = '', added Plot() module function to
# cover most types of plots (lines,images,3D line plots, and volumes), Try it!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 10/29/2025 -> Added "forceGrid" option to PlotLabels() so you can still
# use a grid on mesh/contour plots [but isn't this kind of dumb???]
# Incorporated check for window function, now warns if defaulting to 'hann'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/23/2025 -> Added "cbarfsize" parameter to PlotLabels()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3/18/2025 -> Added CalcHistVsT() method to track time-series' statistical
# properties over time!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2/20/2025 -> Added "squeezeAxes" flag to PlotBispec(), borrowed from
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2/16/2025 -> Fixed missing fwindow input for InstDiffFreq(); finally added
# InstFreqAmp() function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8/16/2024 -> Fixes to PlotInstFreq() [factor of 2pi for zero-cross, ...]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6/17/2024 -> ClickPlot() now autos to CheckNeighbors=True, overplotted red
# lines use alpha=0.5
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6/13/2024 -> Added PlotCoherence() class method; setting InstFreqFlag to 
# "True" (SHIFT+F in PlotGUI...) now allows f_inst vs. t [left click] AND
# |B|/|B|_max vs. \Delta f_inst (ie, inst. diff. freq. vs. amp) [right click];
# TestSignal() now takes inputs for f_samp, t_end, etc.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/23/2024 -> CheckCouple() output enhanced, added "checkdiff" flag
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/16/2024 -> Added 'inst_freq_test' test signal and fixed CheckCouple();
# PlotTrispec() now outputs coordinates of maximum, finally moved colorbar
# from right side to bottom (why not top???); changes to PlotLabels() for
# 3D plots eliminates tough-to-read labels, etc.; fixed issue with window
# function support! Now includes 'hann', 'flat', 'rect', and 'sine'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5/15/2024 -> Fixed PlotTrispec() drawing of max value
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4/24/2024 -> Fixed time vector (self.tv) output for STFT [accounting for
# subint/samprate/2 displacement due to windowing]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4/24/2024 -> Testing flattop window and control of GetBispec(...) phase
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3/20/2024 -> Added WhittakerShannon() for interpolation, debugged beta 
# version of InstFreqZeroCross() and InstDiffFreq()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3/14/2024 -> Finally moved over ApplyBandpass(), ApplyRealBandpass() and
# InstFreq() methods from HP. What better day to debug than pi day?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 11/10/2023 -> Adding "inCOI" attribute to avoid redundant calculations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 10/19/2023 -> Option to plot maximum of TFR ('maxLine' parameter) added to
# PlotSpectro(); fixed oversight with WTrim [was using NFreq instead of 
# len(self.tv)]; new 'TLineCol' attribute to control timeline color; 'COILim'
# attribute controls COI shape; finally fixed ApplySimpleFilter(); PlotRHS()
# method written to simplify twinx() axes
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


# Import dependencies
import os
import sys
import time
# import warnings
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.signal import hilbert
from scipy.special import sici
from scipy.ndimage import uniform_filter1d 

# Computer modern font (LaTeX default)
plt.rcParams['mathtext.fontset'] = 'cm'

class BicAn:
    """Main class of :mod:`pybic` module, i.e., *Bicoherence Analyzer*.

    ``pybic.BicAn(inData,**kwargs)``

    A typical usage is

    .. code-block:: python

        # Import pybic module
        import pybic as bic
        # Create time series to analyze
        x,t,fS = bic.TestSignal('circle')
        # Analyze!
        b = bic.BicAn(x,SampRate=fS)

    Note:
        When using the ``__init__`` method (i.e., ``pybic.BicAn(...)``), keywords are *case insensitive*!
        For example, ``BicAn(inData,samprate=fS)``,  ``BicAn(inData,SAMPRATE=fS)``, and ``BicAn(inData,sAmPrAtE=fS)`` 
        are all valid means of setting the :attr:`SampRate` attribute upon initialization.

        However, the direct augmentation of attributes *is* case sensitive, i.e., ``b.samprate = 100.0`` will throw an error! 

    Args:
        inData (:class:`~numpy.ndarray` or :obj:`str`): Time series to be analyzed **or** :func:`~pybic.TestSignal` string. 
            Using ``'input'`` opens :func:`~pybic.FileDialog` window to choose a local file for input.
        **kwargs: Keyword-argument pairs to set attributes (see :class:`~pybic.BicAn` attributes for more info).

    Returns:
        :class:`~pybic.BicAn`: Output object.

    """
    
    # Attributes
    Date = datetime.now()
    """:class:`~datetime.datetime`:  Date when :mod:`pybic` is loaded."""

    # Private attributes
    _WarnSize  = 1024
    """int: Threshold for large output warning (see :func:`BicAn.SpectroSTFT` and :func:`BicAn.SpectroWavelet`).
    Note that behavior is dictated by :attr:`SizeWarn` attribute."""
    _RunBicAn  = False
    """bool: Internal flag to :class:`~pybic.BicAn` constructor. 
    Suppresses call to :func:`~pybic.BicAn.ProcessData` if False."""
    _NormToNyq = False
    """bool: Internal flag to :func:`~pybic.BicAn.ParseInput`.
    Set to True if :attr:`SampRate` is unset on :class:`~pybic.BicAn` object initialization."""

    Note      = ' '
    """str: User note."""
    Raw       = []
    """:class:`~numpy.ndarray`: Raw input data."""
    Processed = []
    """:class:`~numpy.ndarray`: Processed input data."""
    InstFreq  = []
    """:class:`~numpy.ndarray`: Estimated instantaneous frequency of time-series.
    Note that attribute will remain empty until :func:`BicAn.PlotSpectro` is called (see ``maxLine`` variable)."""

    SampRate  = 1.
    """float: Sampling rate in Hz. 
    Normalizes to Nyquist frequency if unset upon :class:`~pybic.BicAn` object initialization."""
    SubInt    = 512
    """float: STFT subinterval size in samples.
    Note that ``SubInt`` must be smaller than length of raw data, ie, ``len(pybic.Raw)``."""
    Step      = 128
    """float: STFT step in samples.
    Must be less than ``pybic.SubInt``."""
    LimFreq   = 2.
    """float: Maximum frequency of CWT given by ``pybic.SampRate/LimFreq``. 
    Default behavior identical to Nyquist, thus ``LimFreq >= 2``."""
    Window    = 'hann'   
    """str: STFT window function (see :func:`~pybic.HannWindow` and :func:`~pybic.FlatTopWindow`).
    Present support for ``'rect'``, ``'hann'``, ``'sine'``, and ``'flattop'``."""    
    Sigma     = 0.
    """float: CWT time-frequency resolution parameter (see :func:`~pybic.ApplyCWT`). 
    Automatically set to ``np.pi`` if ``Sigma = 0.0``."""
    AlphaExp  = 0.5
    """float: CWT alpha exponent (see :func:`~pybic.ApplyCWT`).
    Default value eliminates frequency dependency of CWT prefactor."""
    CalcCOI   = False
    """bool: Calculate cone of influence (COI) for CWT (see :func:`BicAn.SpectroWavelet`).
    Note that ``CalcCOI = True`` requires an additional CWT calculation."""
    COILim    = -3.0
    """float: Cone of influence logarithmic cutoff (see :func:`BicAn.SpectroWavelet`)."""
    InCOI     = np.array([])
    """:class:`~numpy.ndarray`: Input cone of influence (see :func:`BicAn.SpectroWavelet`)."""
    JustSpec  = False
    """bool: Limit calculation to time-frequency representation (TFR), ie, no polyspectral analysis.
    Note that ``JustSpec = True`` suppresses automatic :func:`BicAn.PlotGUI` call."""
    SpecType  = 'stft'
    """str: Desired time-frequency representation (TFR). 
    Present support for STFT (``'stft'``, ``'fft'``, ``'fourier'``) and CWT (``'cwt'``, ``'wave'``, ``'wavelet'``)."""
    CalcHist  = False
    """bool: Calculate histogram of amplitude from time series (see :func:`~pybic.CalcHistVsT`)."""
    Bispectro = False
    """bool: Calculate bispectrogram. **Not presently implemented!**"""

    ErrLim    = 1e15
    """float: Threshold for mean of time-frequency representation (TFR) over all frequency bins.
    Note that any subintervals which exceed ``ErrLim`` will be set to zero."""
    FScale    = 0
    """int: Log scale for frequency labels (see :func:`~pybic.ScaleToString`).
    For example, ``FScale = 3`` changes frequency units to kHz."""
    TScale    = 0
    """int: Log scale for time labels (see :func:`~pybic.ScaleToString`).
    For example, ``TScale = -3`` changes time units to ms."""
    Filter    = 'none' 
    """str: Desired filter. **Not presently implemented!**"""
    Smooth    = 1  
    """int: Smoothing factor in samples. **Not presently implemented!**"""
    Epsilon   = 1e-6
    """float: Small value in polycoherence calculations."""
    LilGuy    = 1e-6
    """float: Duplicate of BicAn.Epsilon for back-compatability."""
    SizeWarn  = True
    """bool: Display warning for large TFRs."""
    BicVec    = [0,0,0]
    """list: Integers representing desired time series for cross-bispectrum.
    Order dictates ... (see :func:`~pybic.SpecToCrossBispec`)."""
    RandLevel = 1.0
    """float: Level of randomization in bicoherence uncertainty analysis.
    :func:`~pybic.GetBispec` returns true estimate/random phase for ``RandLevel = 0.0``/``RandLevel = 1.0``. 
    See also :func:`~pybic.BicAn.PlotHelper`."""
    DistLevel = 0.0
    """float: Y-axis limit for bicoherence uncertainty analysis.
    Limit set automatically for ``DistLevel = 0.0``."""

    PlotIt    = True
    """bool: Run :func:`BicAn.PlotGUI` after analysis is complete."""
    CMap      = 'viridis'
    """str or :obj:`matplotlib.colors.ListedColormap`: Colormap for plots. 
    Accepts both custom and standard matplotlib colormaps 
    (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)."""
    CbarNorth = True
    """bool: Place colorbar above plot. 
    Setting ``CbarNorth = False`` places colorbar to the "east" of the plot."""
    PlotType  = 'bicoh'
    """str: Desired bispectral quantity to plot (see :func:`BicAn.PlotBispec` and :func:`BicAn.PlotPointOut`).
    Consequently governs the behavior of :func:`BicAn.PlotGUI` and user clicks within.
    Accepts ``'bicoh'``, ``'abs'``, ``'real'``, ``'imag'``, ``'angle'``, ``'mean'``, ``'std'``, and ``'hybrid'``."""
    ScaleAxes = 'manual'
    """str: Desired axis scaling. **Not presently implemented!**"""
    TickLabel = 'normal'
    """str: Tick label customization. **Not presently implemented!**"""
    TLineCol  = 'twilight'
    """str: Colormap for automatic calls to :func:`~pybic.PlotTimeline`."""
    LineWidth = 2
    """int: Plot linewidths in points."""
    FontSize  = 14
    """int: Plot font size in points."""
    PlotDPI   = 150
    """int: Dots per inch of plots."""
    SpecVLim  = []
    """list of float: Spectrogram colorbar limits as ``[vmin,vmax]``.
    Only applies when using :func:`BicAn.PlotGUI`!"""
    NormBic   = False
    """bool: Normalize bicoherence spectrum when using :func:`BicAn.PlotBispec`."""
    PlotSlice = None
    """int: Selected slice of TFR to plot when using :func:`BicAn.PlotPowerSpec` and :func:`BicAn.PlotMeanHist`.
    Also affects output of :func:`BicAn.PlotPowerSpec` and :func:`BicAn.PlotBispec`."""
    PlotSig   = 0
    """int: Selected time series to plot when using :func:`BicAn.PlotSpectro`."""
    BicOfTime = False
    """bool: Calculate bispectrum/bicoherence spectrum for selected :attr:`PlotSlice`."""
    InstFreqFlag = False
    """bool: Toggle behavior of user clicks in :func:`BicAn.PlotGUI`."""

    Detrend   = False
    """bool: Detrend input time series before analysis."""
    ZPad      = False
    """bool: Pad time series with zeros to avoid truncation for STFT."""
    Cross     = False
    """bool: Calculate cross-spectrum/cross-coherence.
    Note that a minimum of 2 input signals are required, i.e., the above does not exist for a single time series"""
    Trispec   = False
    """bool: Calculate trispectrum."""
    Vector    = False
    """bool: Plot linewidths."""
    TZero     = 0.
    """float: Initial time."""

    Figure    = None
    """:obj:`~matplotlib.figure.Figure`: Used for :func:`BicAn.PlotGUI()` functionality."""
    AxHands   = [None,None,None]
    """list of :obj:`~matplotlib.axes.Axes`: List of axes handles."""
    tkVar     = None
    """:obj:`~tkinter.StringVar`: Variable for tkinter."""
    tkRoot    = None
    """:obj:`tkinter`: Root window for tkinter."""
    CaxHands  = [None,None]
    """list of :obj:`~matplotlib.axes.Axes`: List of colorbar axes handles."""
    NewGUICax = False
    """bool: Flag for creating new colorbar axes."""

    tv = []
    """:class:`~numpy.ndarray`: Time vector associated with time-frequency representation"""
    fv = []  
    """:class:`~numpy.ndarray`: Frequency vector associated with time-frequency representation"""
    ff = []   
    """:class:`~numpy.ndarray`: Full frequency vector """

    ft = []  
    """:class:`~numpy.ndarray`: Fourier amplitudes"""
    sg = []   
    """:class:`~numpy.ndarray`: Spectrogram (complex)"""

    cs = []   
    """:class:`~numpy.ndarray`: Cross-spectrum."""
    cc = []  
    """:class:`~numpy.ndarray`: Cross-coherence."""
    cg = []   
    """:class:`~numpy.ndarray`: Coherence spectrum (or 'coherogram')."""

    bs = []   
    """:class:`~numpy.ndarray`: Bispectrum."""
    bc = []   
    """:class:`~numpy.ndarray`: Bicoherence spectrum."""
    bg = []   
    """:class:`~numpy.ndarray`: Bispectrogram."""

    ts = []   
    """:class:`~numpy.ndarray`: Trispectrum (complex)."""
    tc = []   
    """:class:`~numpy.ndarray`: Tricoherence spectrum."""

    er = []   
    """:class:`~numpy.ndarray`: Mean & std dev of FFT."""
    mb = []   
    """:class:`~numpy.ndarray`: Mean bicoherence."""
    sb = []   
    """:class:`~numpy.ndarray`: Std dev of bicoherence spectrum."""

    hg = []   
    """:class:`~numpy.ndarray`: Histogram of input signal amplitudes.
    Note that this is calculated like a spectrogram, i.e., histogram vs. time."""
    mh = []   
    """:class:`~numpy.ndarray`: Input signal histogram averaged over time."""
    ht = []  
    """:class:`~numpy.ndarray`: Time vector of input signal histogram."""
    bv = []  
    """:class:`~numpy.ndarray`: Bin vector of input signal histogram."""

    # Class methods
    def __init__(self,inData,**kwargs):
        """Constructor of :class:`pybic.BicAn`.
        See :class:`~pybic.BicAn` docstring above for more info.
        """
        self.ParseInput(inData,kwargs)

        if self._RunBicAn:
            self.ProcessData()
        return

    def __setattr__(self, attr, val):
        """Set attribute method of :class:`pybic.BicAn`.
        """
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

            if attr=='FScale' and self._NormToNyq:
                val = 0
                print('***NOTE*** :: FScale cannot be set if SampRate is not initialized!')

            self.__dict__[attr] = val

            if attr=='Epsilon':
                self.LilGuy = val
            if attr=='LilGuy':
                self.Epsilon = val
  
        return

    # Dependent properties
    @property
    def FreqRes(self):  
        """float: Frequency resolution in Hz.
        Note that ``FreqRes = SubInt`` if ``SpecType=='cwt'``"""
        return self.SampRate / self.SubInt if self.SpecType=='stft' else self.SampRate / self.Samples

    @property
    def NFreq(self):   
        """int: Number of frequency bins."""
        return int(self.SampRate / self.FreqRes / self.LimFreq)

    @property
    def _Nseries(self): 
        """int: Number of input time series."""
        return min(self.Raw.shape)

    @property
    def Samples(self): 
        """int: Number of samples in processed data."""
        return max(self.Raw.shape) if len(self.Processed)==0 else max(self.Processed.shape)

    @property
    def LineColor(self):
        """str or :obj:`matplotlib.colors.ListedColormap`: Line color in plots.
        See :attr:`CMap` attribute for more info!"""
        if isinstance(self.CMap,mpl.colors.ListedColormap):
            return self.CMap(np.linspace(0,1,256))[:,0:3]
        else:
            return eval('cm.%s( np.linspace(0,1,256) )[:,0:3]' % self.CMap)


    def ParseInput(self,inData,kwargs):
        """Parse inputs for :class:`BicAn` constructor.

        Args:
            inData (:class:`~numpy.ndarray` or :obj:`str`): Time series to be analyzed **or** :func:`~pybic.TestSignal` string. 
                Using ``'input'`` opens :func:`~pybic.FileDialog` window to choose a local file for input.
            kwargs (dict): Keyword-argument pairs to set attributes.
        """
        self._RunBicAn = True  
        print('Checking inputs...') 

        if len(kwargs)==0:
            if isinstance(inData,np.ndarray):
                # If array input, use normalized frequencies  
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
                            'quad_couple_circle2','inst_freq_test','linear_phase','phase_mod','linear_phase_am','phase_mod_am',
                            '3tone_short','circle_oversample','cross_3tone_short','helix']
                if instr == 'input':
                    # Start getfile prompt
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
                    if sz[1] > sz[0]:                 # Check row vector
                        inData = np.transpose(inData) # Transpose if so

                    # Check if first dimension is strictly increasing
                    if np.sum(np.sign(np.diff(inData[:,0]))) == N-1:
                        print('Time input detected!') 
                        self.Raw   = np.zeros((N,min(sz)-1))
                        self.TZero = inData[0,0]
                        self.Raw   = inData[:,1:]
                    else:
                        self.Raw = np.zeros((N,min(sz)))
                        self.Raw = inData   
                
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

            self.Step = int(abs(self.Step))                # Remove sign and decimals
            if self.Step==0 or self.Step>self.SubInt:      # Check step <= subinterval
                self.Step = self.SubInt//4                 # This seems fine?
                print('***WARNING*** :: Step must be nonzero and less than subint... Using {}.'.format(self.Step))     

        return


    def ApplyZPad(self):
        """Apply zero padding to :attr:`Raw` data.

        Truncates data when :attr:`ZPad` is ``False``.

        .. caution::

            This should only be applied if using the STFT, i.e.,
            when :attr:`SpecType` corresponds to ``stft``!

        """
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
        """Main processing loop (see source code).
        """
        start = time.time()
 
        dum = self.SpecType.lower()
        if dum in ['fft', 'stft', 'fourier']:
            self.ApplyZPad()
            self.SpectroSTFT()
            self.SpecType = 'stft'
        elif dum in ['wave', 'wavelet', 'cwt']:
            self.SpectroWavelet()
            self.SpecType = 'wave'   

        if self._RunBicAn:
            if self.CalcHist:
                self.HistogramSig()    
            if self.Cross:
                self.Coherence()
            if not self.JustSpec:
                self.Bicoherence()
                if self.Trispec:
                    self.Tricoherence()
            ##################
            end = time.time()
            print('Complete! Process required %.5f s.' % (end-start))
    
            if self.PlotIt and not self.JustSpec:      
                self.PlotGUI()


    ## Analysis
    def SpectroSTFT(self):
        """Class wrapper for :func:`ApplySTFT`.
        """
        if self.NFreq>self._WarnSize and self.SizeWarn:
            self.SizeWarnPrompt(self.NFreq)

        if self._RunBicAn:
            spec,afft,f,t,err,Ntoss = ApplySTFT(self.Processed,self.SampRate,self.SubInt,self.Step,self.NFreq,self.TZero,self.Detrend,self.ErrLim,self.Window)
            
            self.tv = t
            self.fv = f

            self.ft = afft

            self.sg = spec
            self.er = err     
        return  


    def SpectroWavelet(self):
        """Class method for CWT analysis.

        .. note::

            This method *mostly* wraps :func:`ApplyCWT`, in addition to

            * Checking the default value of :attr:`Sigma`,
            * Use of :func:`ApplyDetrend` if :attr:`Detrend` is ``True``,
            * Subtraction of time-series' mean.

        """
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

        if self._RunBicAn:
            CWT,acwt,f,t = ApplyCWT(self.Processed,self.SampRate,self.Sigma,self.LimFreq,self.AlphaExp)

            if self.CalcCOI:
                if len(self.InCOI)==0:
                    nz = np.zeros((len(self.Processed),1))
                    nz[0,:] = 1
                    nz[-1,:] = 1
                    self.InCOI,_,_,_ = ApplyCWT(nz,self.SampRate,self.Sigma,self.LimFreq,self.AlphaExp)
                for k in range(self._Nseries):
                    coiMask = ( (abs(self.InCOI)/np.max(abs(self.InCOI)) ) < np.exp(self.COILim) )
                    CWT[:,:,k] = CWT[:,:,k] * coiMask[:,:,0]
                    acwt[:,k]  = np.mean(abs(CWT[:,:,k])**2,1)

                    #CWT[:,:,k] = nzCWT[:,:,k]

            self.tv = t + self.TZero
            self.fv = f
            self.ft = acwt 
            self.sg = CWT
        return


    def HistogramSig(self,Nbins=200):
        """Class wrapper for :func:`CalcHistVsT`.

        Args:
            Nbins (int): Number of histogram bins.
        """
        binMax = np.max(abs(self.Processed))
        hist,mh,binvec,time_vec = CalcHistVsT(self.Processed,self.SampRate,self.SubInt,self.Step,self.TZero,binMax=binMax,Nbins=Nbins)
        
        self.hg = hist
        self.mh = mh

        self.bv = binvec
        self.ht = time_vec
        return  


    def Coherence(self):
        """Class wrapper for :func:`SpecToCoherence`.
        """
        if self._Nseries!=2:
            print('***WARNING*** :: Cross-coherence requires exactly 2 signals!')
        else:
            cspec,crosscoh,coh = SpecToCoherence(self.sg,self.LilGuy)
            self.cs = cspec      # Cross-spectrum
            self.cc = crosscoh   # Cross-coherence
            self.cg = coh        # Cohero-gram
        return


    def Bicoherence(self):
        r"""Class wrapper for bispectral analysis.

        .. note::

            For a single time-series :math:`x[t]`, this method wraps :func:`SpecToBispec`,
            producing the **auto**-bispectrum/-bicoherence spectrum,
            :math:`\mathcal{B}_{xxx}`/:math:`b^2_{xxx}`.

            :func:`SpecToCrossBispec` is wrapped for 2 or 3 time-series,
            yielding the **cross**-bispectrum/-bicoherence spectrum,
            :math:`\mathcal{B}_{xyy}`/:math:`b^2_{xyy}` or
            :math:`\mathcal{B}_{xyz}`/:math:`b^2_{xyz}`.

        .. caution::

            This method omits the first and last 10% of the spectrogram :attr:`sg`
            when using the CWT (:attr:`SpecType` = ``cwt``) to reduce edge effects!
            This can be avoided by setting :attr:`CalcCOI` to ``True``.

        """    
        dum = self.sg 
        if self.SpecType == 'wave' and not self.CalcCOI:
            WTrim = len(self.tv) // 10
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
        """Class wrapper for trispectral analysis (:func:`SpecToTrispec`).

        .. note::

            Only the **auto**-trispectrum/-tricoherence spectrum is presently supported!

        .. caution::

            This method omits the first and last 10% of the spectrogram :attr:`sg`
            when using the CWT (:attr:`SpecType` = ``cwt``) to reduce edge effects!
            This can be avoided by setting :attr:`CalcCOI` to ``True``.

        """       
        dum = self.sg 
        if self.SpecType == 'wave' and not self.CalcCOI:
            WTrim = len(self.tv) // 10
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
        """Calculate mean of bicoherence spectrum across the full bi-frequency space.

        Uses absolute value of spectrogram :attr:`sg` and randomized phases.

        Args:
            Ntrials (int): Number of randomized trials.
        """ 
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


    def MonteCarloMax(self,N=2,Nrolls=1000,critCoh=1.0,plot=False,verbose=False):
        """Identifies maxima in N-coherence spectra with random restart hillclimb.

        Uses absolute value of spectrogram :attr:`sg` and randomized phases.

        Args:
            N (int): Polyspectral order.
            Nrolls (int): Number of randomized trials.
            critCoh (float): Critical level of ``N``-coherence.
            plot (bool): Plot data (``N`` must be ``2`` or ``3``).
            verbose (bool): Show tested frequencies.
        """  
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
        """Plots time average (or slice) of spectrogram.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            vLim  (list): Logarithmic y-limits of plot ``[vmin,vmax]``, defaults to data range.

        .. note::

            The :attr:`PlotSlice` attribute allows plotting a slice of spectrogram ``sg[:,PlotSlice,:]``.
            Defaults to time average when ``None``.

        """
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
        else:
            fig = args[0]
            ax  = args[1]

        f = self.fv/10**self.FScale
        dum = self.ft if self.PlotSlice is None else abs(self.sg[:,self.PlotSlice,:])**2

        for k in range(self._Nseries):
            ax.semilogy(f, dum[:,k], linewidth=self.LineWidth, color=self.LineColor[(50+40*k) % 256])

        fstr = r'$f/f_s$' if self._NormToNyq else r'$f\,\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        ystr = r'$\langle|%s(f)|^2\rangle\,\mathrm{[arb.]}$' % (r'\mathcal{X}' if self.SpecType=='stft' else r'\mathcal{W}')
        PlotLabels(fig,ax,[fstr,ystr],self.FontSize,self.CbarNorth)
        ax.set_xlim(f[0], f[-1])
        if len(vLim)==2:
            ax.set_ylim(10**vLim[0],10**vLim[1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return

    def PlotMeanHist(self,*args,vLim=[]):
        """Plots time average (or slice) of data histogram.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            vLim  (list): Logarithmic y-limits of plot ``[vmin,vmax]``, defaults to data range.

        .. note::

            The :attr:`PlotSlice` attribute allows plotting a slice of histogram ``hg[:,PlotSlice,:]``.
            Defaults to time average when ``None``.

        """
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
        else:
            fig = args[0]
            ax  = args[1]

        b = self.bv
        dum = self.mh if self.PlotSlice is None else self.hg[:,self.PlotSlice,:]

        for k in range(self._Nseries):
            ax.semilogy(b, 100*dum[:,k], linewidth=self.LineWidth, color=self.LineColor[(50+40*k) % 256])

        bstr = r'${\rm Amp.}\, [\mathrm{arb.}]$'
        ystr = r'${\rm \%}$'
        PlotLabels(fig,ax,[bstr,ystr],self.FontSize,self.CbarNorth)
        ax.set_xlim(b[0], b[-1])
        if len(vLim)==2:
            ax.set_ylim(10**vLim[0],10**vLim[1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotCoherence(self,*args,crossSpec=False,vLim=[]):
        """Plots cross-coherence spectrum :attr:`cc` or cross-spectrum :attr:`cs`.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            crossSpec (bool): Plots absolute value of cross-spectrum if ``True``.
            vLim  (list): Logarithmic y-limits of plot ``[vmin,vmax]``, defaults to data range.
        """
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
        else:
            fig = args[0]
            ax  = args[1]

        f = self.fv/10**self.FScale

        dum = np.mean(abs(self.cs)**2,1) if crossSpec else self.cc
        if crossSpec:
            ax.semilogy(f, np.mean(abs(self.cs)**2,1), linewidth=self.LineWidth, color=self.LineColor[50])
        else:
            ax.plot(f, self.cc, linewidth=self.LineWidth, color=self.LineColor[50])

        fstr = r'$f/f_s$' if self._NormToNyq else r'$f\,\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        ystr = r'$\langle|\mathcal{C}(f)|^2\rangle\,{\rm [arb.]}$' if crossSpec else r'$c^2(f)$'
        PlotLabels(fig,ax,[fstr,ystr],self.FontSize,self.CbarNorth)
        ax.set_xlim(f[0], f[-1])
        if len(vLim)==2:
            ax.set_ylim(10**vLim[0],10**vLim[1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotSpectro(self,*args,vLim=[],maxLine=-1.):
        """Plots absolute value of spectrogram.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            vLim  (list): Logarithmic y-limits of plot ``[vmin,vmax]``, defaults to data range.
            maxLine (float): Overplots peak of spectrogram vs time if ``maxLine>=0``.
                Nonnegative input also populates the :attr:`InstFreq` attribute!
        """
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
            cax = None
        else:
            fig = args[0]
            ax  = args[1]
            cax = self.CaxHands[1]

        tstr = r'$f/T$' if self._NormToNyq else r'$t\, [\mathrm{%ss}]$' % (ScaleToString(self.TScale))
        fstr = r'$f/f_s$' if self._NormToNyq else r'$f\,\, [\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        cbarstr = r'$\log_{10}|%s(t,f)|^2$' % (r'\mathcal{X}' if self.SpecType=='stft' else r'\mathcal{W}')

        t = self.tv/self.tv[-1] if self._NormToNyq else self.tv/10**self.TScale
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

        if maxLine>=0:
            _,Nf = arrmin( abs( f - maxLine ) )
            
            finst = 0*self.tv
            for k in range(len(self.tv)):
                _,m = arrmin( -abs(self.sg[Nf:,k,self.PlotSig]))
                finst[k] = self.fv[Nf+m]

            self.InstFreq = finst
            ax.plot(t, finst / 10**self.FScale, color='gray')

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotHisto(self,*args,vLim=[]):
        """Plots data histogram vs time.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            vLim  (list): Logarithmic y-limits of plot ``[vmin,vmax]``, defaults to data range.
        """
        if len(args)==0:
            fig, ax = plt.subplots(dpi=self.PlotDPI)
            cax = None
        else:
            fig = args[0]
            ax  = args[1]
            cax = self.CaxHands[1]

        tstr = r'$f/T$' if self._NormToNyq else r'$t\, [\mathrm{%ss}]$' % (ScaleToString(self.TScale))
        bstr = r'${\rm Amp.}\, [\mathrm{arb.}]$'
        cbarstr = r'$\log_{10}\left({\rm \%}\right)$'

        t = self.ht/self.ht[-1] if self._NormToNyq else self.ht/10**self.TScale
        b = self.bv

        if len(vLim)==2:
            im = ax.pcolormesh(t,b,np.log10(100*self.hg[:,:,self.PlotSig] + 0*1e-16), cmap=self.CMap, shading='auto', vmin=vLim[0], vmax=vLim[1])
        else:
            im = ax.pcolormesh(t,b,np.log10(100*self.hg[:,:,self.PlotSig] + 0*1e-16), cmap=self.CMap, shading='auto')
        cax = PlotLabels(fig,ax,[tstr,bstr,cbarstr],self.FontSize,self.CbarNorth,im,cax)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(b[0], b[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return


    def PlotBispec(self,*args,normb2=False,plot3d=False,squeezeAxes=True):
        """Plots bispectrum or bicoherence spectrum.

        Args:
            *args (list): Input :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes` as ``[fig,ax]``.
                Default behavior creates new figure and axes with :func:`~matplotlib.pyplot.subplots`.
            normb2 (bool): Normalize colorbar limits to ``[0,1]``.
            plot3d (bool): Plot in 3D with :func:`~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`
                (default behavior uses :func:`~matplotlib.pyplot.pcolormesh`). 
            squeezeAxes (bool): Shrink y-axis to eliminate white space in plot.

        .. note::

            The output of ``PlotBispec()`` depends on the :attr:`PlotType` attribute!
            For example, ``PlotType = 'bicoh'`` provides the bicoherence spectrum,
            while ``PlotType = 'angle'`` plots the biphase spectrum.

        """
        if len(args)==0:

            if plot3d:
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'},dpi=self.PlotDPI)
            else:
                fig, ax = plt.subplots(dpi=self.PlotDPI)

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
                # EXPERIMENTAL 3D OPTION!!!
                X,Y = np.meshgrid(f, f[0:len(f)//2] )
                im = ax.plot_surface(X, Y, dum, cmap=self.CMap, lw=0, antialiased=False)
            else:
                if normb2:
                    im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto', vmin=0, vmax=1)
                else:
                    im = ax.pcolormesh(f,f[0:len(f)//2],dum, cmap=self.CMap, shading='auto')

                # Draw triangle
                ax.plot([0, f[-1]/2],[0, f[-1]/2],     color=[0.5,0.5,0.5], linewidth=2.5)
                ax.plot([f[-1]/2, f[-1]],[f[-1]/2, 0], color=[0.5,0.5,0.5], linewidth=2.5)

            if squeezeAxes:
                ax.set_ylim(f[0], f[-1]/2)
            else:
                ax.set_ylim(f[0], f[-1])

        else:
            f = self.ff/10**self.FScale
            if normb2:
                im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto', vmin=0, vmax=1)
            else:
                im = ax.pcolormesh(f,f,dum, cmap=self.CMap, shading='auto')
            ax.set_ylim(f[0], f[-1])
        
        fstr1 = r'$f_1/f_s$' if self._NormToNyq else r'$f_1\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr2 = r'$f_2/f_s$' if self._NormToNyq else r'$f_2\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        strings = [fstr1,fstr2,cbarstr,''] if plot3d else [fstr1,fstr2,cbarstr] 
        cax = PlotLabels(fig,ax,strings,self.FontSize,self.CbarNorth,im,cax)
        if self.NewGUICax:
            self.CaxHands[0] = cax
        ax.set_xlim(f[0], f[-1])

        if len(args)==0:
            plt.tight_layout()
            plt.show()
        return      


    def PlotTrispec(self,Tval=0.5,colorTricoh=True,elev=26,azim=-56,roll=0,shrink=0.7,squeezeAxes=True):
        """Plots trispectrum or tricoherence spectrum.

        Args:
            Tval (float): Critical value of tricoherence (omits all data below this value).
            colorTricoh (bool): Color data with tricoherence (``True``) or triphase (``False``).
            elev (float): Viewpoint angle above xy axes.
            azim (float): Viewpoint azimuth around z axis.
            roll (float): Viewpoint roll angle.
            shrink (float): Shrink factor of colorbar.
            squeezeAxes (bool): Shrink y/z-axes to eliminate white space in plot.

        Returns:
            list of int: Coordinates of maximum value of tricoherence.
        """
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
        im = ax.scatter(x[q],y[q],z[q],c=dum,cmap=self.CMap,alpha=0.5,marker='o')

        ax.view_init(elev=elev, azim=azim, roll=roll)

        if squeezeAxes:
            ax.set_xlim(0,f[-1]) 
            ax.set_ylim(0,f[-1]/2)
            ax.set_zlim(0,f[-1]/3)
        else:
            ax.set_xlim(0,f[-1]) 
            ax.set_ylim(0,f[-1])
            ax.set_zlim(0,f[-1])

        fstr1 = r'$f_1/f_s$' if self._NormToNyq else r'$f_1\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr2 = r'$f_2/f_s$' if self._NormToNyq else r'$f_2\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        fstr3 = r'$f_3/f_s$' if self._NormToNyq else r'$f_3\,[\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        cax = PlotLabels(fig,ax,[fstr1,fstr2,fstr3,cbarstr],self.FontSize,self.CbarNorth,im,None,shrink=shrink)

        # Find maximum
        max_ind = np.unravel_index(np.argmax(self.tc),self.tc.shape)
        print('@ (fx,fy,fz) = ',f[max_ind[1]],f[max_ind[0]],f[max_ind[2]])
        # "Paint axes"
        fx = f[max_ind[1]]
        fy = f[max_ind[0]]
        fz = f[max_ind[2]]

        greenLim = lim2 if squeezeAxes else -1
        plt.plot([fx,fx], [fy,f[greenLim]], [fz,fz],'-g.',lw=1.25)
        plt.plot([fx,fx], [fy,fy], [0,fz],'-b.',lw=1.25)
        plt.plot([0,fx], [fy,fy],[fz,fz],'-r.',lw=1.25)

        DrawSimplex(f[-1])

        plt.plot([0,fx],[0,0],[0,0],'-r',lw=1.25)
        plt.plot([0,0],[0,fy],[0,0],'-g',lw=1.25)
        plt.plot([0,0],[0,0],[0,fz],'-b',lw=1.25)
        
        plt.tight_layout()
        plt.show()
        
        return [max_ind[1],max_ind[0],max_ind[2]]


    def WhichPlot(self,local=None):
        """Assists plotting functions with data and labeling.

        Args:
            local (:class:`~numpy.ndarray`): Local bispectrum to plot.

        Returns:
            list: ``dum,cbarstr = WhichPlot(...)``

            * dum (:class:`~numpy.ndarray`) - Raw data of plottable.
            * cbarstr (:obj:`str`) - Axis label for chosen plottable.

        .. note::

            Output strings include tildes when ``local`` is not ``None``!

        """
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


    def PlotHelper(self,whatPlot,X,Y,IsFreq=False,CheckNeighbors=False,fig=None,ax=None,Ntrials=200,b2bins=100,cVal=0.999,SaveAs=None,NoXLabel=False):
        """Estimate and plot distribution of bispectrum/bicoherence for single point.

        Args:
            whatPlot (str): Input string indicating desired plot.
                Accepts ``'b2Prob'``, ``'Phasor'``, or ``'BvsTime'``. 
            X (list): Frequency 1 (:attr:`fv` bin or frequency).
            Y (list): Frequency 2 (:attr:`fv` bin or frequency).
            IsFreq (bool): ``True`` for ``X`` and ``Y`` in frequency units.
            CheckNeighbors (bool): Perform analysis on nearby points.
            fig (:obj:`~matplotlib.figure.Figure`): Input figure.
            ax (:obj:`~matplotlib.axes.Axes`): Input axes.
            Ntrials (int): Number of random trials for uncertainty analysis (``whatPlot='b2Prob'``).
            b2bins (int): Number of bicoherence bins in uncertainty analysis (``whatPlot='b2Prob'``).
            cVal (float): Critical level for random phase b2 estimate (``whatPlot='b2Prob'``).
            SaveAs (str): Name of output file (if not :obj:`None`.)
            NoXLabel (bool): Omit x axis label.

        Returns:
            list: Output of uncertainty analysis (i.e., empty list if ``whatPlot!='b2Prob'``).

            ``[<bootstrap mean>, <bootstrap std>, <critical b2>, <noise floor>]``

        .. caution::

            When ``IsFreq = True``, the corresponding frequency bin is 
            dependent on the :attr:`FScale` attribute, i.e., 
            ``X=[6],Y=[7]`` indicates 6 and 7 **kHz** for ``FScale=3``
            but 6 and 7 **nHz** for ``FScale=-9``.

        .. seealso::

            :func:`BicAn.PlotPointOut`
                Wrapper function with simpler interface.

            :func:`BicAn.PlotGUI`
                Graphical interface which subsumes :func:`BicAn.PlotPointOut` and :func:`BicAn.PlotHelper`.

        """
        out = []
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
            # X = [X[0],X[0],X[0]-1,X[0]+1]
            # Y = [Y[0],Y[0]+1,Y[0],Y[0]]
            # For demo figure
            X = [X[0],X[0]-1,X[0]+1,X[0]+2]
            Y = [Y[0],Y[0],Y[0],Y[0]]
            self.PlotHelper(whatPlot=whatPlot,X=X,Y=Y,IsFreq=False,CheckNeighbors=CheckNeighbors,fig=fig,ax=ax,
                Ntrials=Ntrials,b2bins=b2bins,cVal=cVal)
            return
        
        if whatPlot=='b2Prob':

            g = np.zeros((Ntrials))

            print('Calculating distribution for {}...      '.format(pntstr[0]))
            if self.SpecType=='wave':
                WTrim = len(self.tv) // 10
                dum = self.sg[:,WTrim:-WTrim,:] 
            else:
                dum = self.sg
            for k in range(Ntrials):
                LoadBar(k,Ntrials)
                g[k],_,_ = GetBispec(dum,self.BicVec,self.LilGuy,Y[0],X[0],self.RandLevel)

            # Limit b^2, create vector, and produce histogram 
            b2lim  = 1 
            b2vec  = np.linspace(0,b2lim,b2bins)
            cnt,_  = np.histogram(g, bins=b2bins, range=(0,b2lim) )

            # Integrate count
            intcnt = sum(cnt) * ( b2vec[1] - b2vec[0] )
            # exp dist -> (1/m)exp(-x/m)
            m = np.mean(g)
            ax.plot(b2vec, cnt/intcnt, linewidth=self.LineWidth, color=self.LineColor[50], marker='x', linestyle='none', label='Randomized')
            # More accurate distribution... Just more complicated! (Get to it later...)
            #semilogy(b2vec,(1/m)*exp(-b2vec/m).*(1-b2vec),'linewidth',self.LineWidth,'color','red'); 
            b2dist = (1/m)*np.exp(-b2vec/m)
            ax.plot(b2vec, b2dist, linewidth=self.LineWidth, label=r'$(1/\mu)e^{-b^2/\mu}$')

            # Get bootstrap estimate
            b2boot,_,_ = GetBispecBootstrap(dum,self.BicVec,self.LilGuy,Y[0],X[0],Ntrials=Ntrials)
            cntboot,_  = np.histogram(b2boot, bins=b2bins, range=(0,b2lim) )
            intcntboot = sum(cntboot) * ( b2vec[1] - b2vec[0] )
            ax.plot(b2vec, cntboot/intcntboot,':',linewidth=self.LineWidth/2, color='C2', label=r'${\rm PDF}_{\rm bootstrap}$')

            ##N = len(self.tv)
            N = dum.shape[1]
            # Maybe I misunderstand things?
            # chi2dist = np.exp(-b2vec/N) / 2
            # ax.plot(b2vec, chi2dist, linewidth=self.LineWidth, label=r'$(1/2N)\chi^2_2$')
            # This should be chi-squared in general....
            # from scipy.special import gamma
            # X2 = lambda x,nu: x**(nu/2-1) * np.exp(-x/2) / ( 2**(nu/2) * gamma(nu/2) )
            print('m = %f\nvar = %f\nnu = %f' % (m,np.var(g),2 * m**2/np.var(g)) )

            b2crit = -m*np.log(1 - cVal)
            b2true,_,_ = GetBispec(dum,self.BicVec,self.LilGuy,Y[0],X[0],False)
            
            b2noisebase = 10 if self.SpecType=='stft' else self.SampRate / min( abs(self.fv[X[0]]), abs(self.fv[Y[0]]), abs(self.fv[X[0]+Y[0]]) ) # vanMilligen PoP (1995)
            # 10 from Fackrell (1995), ElgarGuza (1988)
            # Actually, 9.2 represents the 99% confidence level, so 10 is >99%
            b2noise = ( b2noisebase / (2*N) )**0.5
            errb2 = N**(-0.5)
            # errb2 = (2 * b2true/N * (1-b2true)**3 )**0.5 ## <- This is from ElgarSebert (1989)
            ax.axvspan(b2true-errb2, b2true+errb2, facecolor='C2', alpha=0.2)

            print('1/sqrt(N) = %f\nstd(rand) = %f\nstd(boot) = %f' % (errb2,np.var(g)**0.5,np.var(b2boot)**0.5) )

            # Plot bootstrap error
            ax.axvspan(np.mean(b2boot)-np.var(b2boot)**0.5, np.mean(b2boot)+np.var(b2boot)**0.5, facecolor='C2', alpha=0.2)
            # ax.axvline(np.mean(b2boot)-np.var(b2boot)**0.5,linestyle=':', linewidth=self.LineWidth/2, color='C2')
            # ax.axvline(np.mean(b2boot)+np.var(b2boot)**0.5,linestyle=':', linewidth=self.LineWidth/2, color='C2')

            # Calculated standard deviation
            ax.axvspan(b2true-(np.var(g))**0.5, b2true+(np.var(g))**0.5, facecolor='C2', alpha=0.2)

            yrange = [0, b2dist[0]] if self.DistLevel==0 else [0,self.DistLevel]
            #yrange = [1e-3, b2dist[0]*10]
            ax.axvline(b2crit, linewidth=self.LineWidth, color='C1', label=r'$99.9\%$ CL')
            ax.axvline(b2true, linewidth=self.LineWidth, color='C2', label='Measured',)
            ax.axvline(b2noise, linewidth=self.LineWidth, color='C3', label='Noise floor')

            PlotLabels(fig,ax,['$b^2$' + pntstr[0],r'$\mathrm{PDF}$'], self.FontSize, self.CbarNorth)

            ax.set_xlim(0,1)
            ax.set_ylim(yrange[0],yrange[1])

            out = [np.mean(b2boot),np.var(b2boot)**0.5,b2crit,b2noise]

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
            tstr = ' ' if NoXLabel else r'$t\, [\mathrm{%ss}]$' % ( ScaleToString(self.TScale) )
            PlotLabels(fig,ax,[tstr,ystr],self.FontSize,self.CbarNorth)

        elif whatPlot=='Phasor':

            _,_,Bi = GetBispec(self.sg,self.BicVec,self.LilGuy,Y[0],X[0],False)
            Bi /= np.max(abs(Bi))

            t = self.tv/10**self.TScale
            tstr = r'$t\, [\mathrm{%ss}]$' % ScaleToString(self.TScale)
            PlotTimeline(np.real(Bi),np.imag(Bi),t=t,fig=fig,ax=ax,lw=self.LineWidth,cmap=self.TLineCol,cbar=tstr)
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
        return out


    def PlotPointOut(self,X,Y,IsFreq=False,PlotAll=False,SaveAs=None,CheckNeighbors=False,Ntrials=200):
        """Wraps :func:`BicAn.PlotHelper` for :func:`BicAn.PlotGUI`.

        Args:
            X (list): Frequency 1 (:attr:`ft` bin or frequency).
            Y (list): Frequency 2 (:attr:`ft` bin or frequency).
            IsFreq (bool): ``True`` for ``X`` and ``Y`` in frequency units.
            PlotAll (bool): Plot all available facets of bispectral analysis.
            SaveAs (str): Name of output file (if not :obj:`None`).
            CheckNeighbors (bool): Perform analysis on nearby points.
            Ntrials (int): Number of random trials for uncertainty analysis.

        Returns:
            list of int: Indices of :attr:`fv` corresponding to ``X``, ``Y``.
        """
        if PlotAll: 

            old_plotType = self.PlotType

            fig = plt.figure(dpi=self.PlotDPI)

            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)

            # print('input is...',X,Y)
            self.PlotHelper('b2Prob',fig=fig,ax=ax1,X=X,Y=Y,IsFreq=IsFreq,CheckNeighbors=CheckNeighbors,Ntrials=Ntrials)

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
            self.PlotHelper('b2Prob',X=X,Y=Y,IsFreq=IsFreq,SaveAs=SaveAs,Ntrials=Ntrials)

        elif self.PlotType in ['abs','real','imag','angle']:
            self.PlotHelper('BvsTime',X=X,Y=Y,IsFreq=IsFreq,SaveAs=SaveAs,CheckNeighbors=CheckNeighbors)

        return X,Y


    def RefreshGUI(self,SaveAs=None):
        """Refreshes GUI as initiated by :func:`BicAn.PlotGUI`.

        Args:
            SaveAs (str): Name of output file (if not :obj:`None`).
        """
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
        if SaveAs is None:
            fig.canvas.draw()
            plt.show()
            # Swapped this out after reading https://stackoverflow.com/questions/30880358/matplotlib-figure-not-updating-on-data-change
        else:
            fig.savefig(SaveAs,dpi=self.PlotDPI,bbox_inches='tight')
            plt.close(fig)
        self.NewGUICax = False
        return


    def PlotGUI(self,SaveAs=None,subplotType=None):
        """Main graphical interface for exploring :class:`BicAn` data.

        Args:
            SaveAs (str): Name of output file (if not :obj:`None`).
            subplotType (str): Swap bispectrum anf spectrogram (if not :obj:`None`).
        """
        fig = plt.figure(dpi=self.PlotDPI) ###,figsize=[9,6])

        if subplotType is None:
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(224)
        else:
            # Alternate joint ~> Should have option, yes?
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(122)
            ax3 = plt.subplot(223)

        # Save figure and axes with object
        self.Figure = fig
        self.AxHands = [ax1, ax2, ax3]
        self.CaxHands = [None,None]

        cid = fig.canvas.mpl_connect('button_press_event', self.ClickPlot)
        pid = fig.canvas.mpl_connect('key_press_event', self.SwitchPlot)
        
        self.NewGUICax = True
        self.RefreshGUI(SaveAs=SaveAs)
        return


    def FindMaxInRange(self,FxLo,FxHi,FyLo,FyHi,useb2=True):
        r"""Finds maximum bicoherence in given range.

        Helpful for tracking a particular feature over time.

        Args:
            FxLo (float): Low (x-)frequency limit.
            FxHi (float): High (x-)frequency limit.
            FyLo (float): Low (y-)frequency limit.
            FyHi (float): High (y-)frequency limit.
            useb2 (bool): Use max bicoherence.
                Finds max bispectral modulus :math:`|\mathcal{B}|` if ``False``.

        .. caution::

            This method does not work properly with **cross**-bispectral 
            analysis at present!

        """
        # Transform to desired scaling
        dum = self.fv / 10**self.FScale
        vx = (dum>FxLo) * (dum<FxHi) * 1
        vy = (dum>FyLo) * (dum<FyHi) * 1
        
        vy = vy[0:len(vy)//2]
        
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


    def SizeWarnPrompt(self,n):
        """Modal warning about large FFT size.

        Args:
            n (int): Requested size.
        """
        qwer = messagebox.askokcancel('Question',f'FFT elements exceed {self._WarnSize}! ({n}) Continue?')
        if not qwer:
            print('Operation terminated by user.')
            self._RunBicAn = False
        #qwer.withdraw()
        #qwer.destroy()
        return         # Bail if that seems scary! 


    def ClickPlot(self,event):
        """Callback for :func:`BicAn.PlotGUI` clicks.

        Args:
            event (:obj:`~matplotlib.backend_bases.MouseEvent`): Click event object.
        """
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
                    self.AxHands[1].axhline(f[k], color='red', linewidth=1.5,alpha=0.5)
                    self.AxHands[2].axvline(f[k], color='red', linewidth=1.5,alpha=0.5)

            # Aha! This is the secret sauce!
            self.Figure.canvas.draw()

            # 1 = MouseButton.LEFT
            # 3 = MouseButton.RIGHT
            if event.button==1:
                self.PlotPointOut([Ix],[Iy]) if not self.InstFreqFlag else self.PlotInstFreq(Ix,Iy)
            elif event.button==3:
                self.PlotPointOut([Ix],[Iy],PlotAll=True,CheckNeighbors=True) if not self.InstFreqFlag else self.InstDiffFreq(Ix,Iy,histo=True)

        elif ax == self.AxHands[1]: # Check spectrogram
            tx = event.xdata
            t  = self.tv/10**self.TScale
            _,self.PlotSlice = arrmin( abs(t-tx) ) # Find closest point in time
            self.RefreshGUI()

        else:
            print('No callback there!')
        return    


    def SwitchPlot(self,event):
        """Callback for :func:`BicAn.PlotGUI` keypresses.

        Args:
            event (:obj:`~matplotlib.backend_bases.KeyEvent`): Key event object.
        """
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
            print('Some kind of help menu here!')
        elif key == 'X': # Reset GUI
            self.PlotSlice = None
            self.BicOfTime = False
            self.PlotType = 'bicoh'
        elif key == 'T':
            self.BicOfTime = not self.BicOfTime
        elif key == 'F':
            self.InstFreqFlag = not self.InstFreqFlag
        elif key == 'C':
            self.SwitchCMap()
        elif key == 'right':
            self.PlotSlice = 0 if self.PlotSlice is None else (self.PlotSlice + 10) % len(self.tv) 
        elif key == 'left':
            self.PlotSlice = len(self.tv)-1 if self.PlotSlice is None else (self.PlotSlice - 10) % len(self.tv)
        else:
            return

        # Activate!
        self.RefreshGUI()
        return


    def SwitchCMap(self):
        """Interactive colormap selection.

        See :func:`BicAn.SwitchPlot` for more info!

        """

        self.tkRoot = tk.Tk()
        self.tkVar = tk.StringVar(value='viridis',master=self.tkRoot)

        cmaps = ['viridis','gnuplot2','PiYG']

        tk.Label(self.tkRoot, 
                 text = 'Pick a colormap!',
                 justify = tk.LEFT,
                 padx = 20).pack()

        for cmap in cmaps:
            tk.Radiobutton(self.tkRoot, 
                           text = cmap,
                           padx = 20, 
                           variable = self.tkVar, 
                           command = self.SwitchCMapClick,
                           value = cmap
                           ).pack(anchor=tk.W)
        return


    def SwitchCMapClick(self):
        """Callback function for :func:`BicAn.SwitchCMap`."""

        self.CMap = self.tkVar.get()
        self.tkRoot.destroy()
        self.RefreshGUI()
        return


    def CheckCouple(self,f,checkdiff=False):
        """Check nth order coupling for a given test vector of freqs. 

        Args:
            f (list): List of frequency bins.
            checkdiff (bool): Check for difference frequency coupling.

        """
        f = f if self._Nseries==1 else np.array(f) - len(self.fv)

        n = len(f)
        mask = bin_mat(n)
        L = 2**(n-1)
        out = np.zeros(L)
        for k in range(L):
            dum = f * mask[k,:]
            if checkdiff:
                #dum[0] = f[0] + np.sum(-dum[1::])
                dum[0] = sum(f)
            try:
                out[k],_,_ = GetPolySpec(self.sg,dum,self.LilGuy)
            except:
                print(f'Whoops... the requested sum bin (# {sum(dum)}) is greater than NFreq ({self.NFreq})!')
            print(dum,'=',self.fv[1]*dum/10**self.FScale, '%sHz' % (ScaleToString(self.TScale)),' ~>',out[k])
        print('Mean is ',np.mean(out))
        return out


    def InstDiffFreq(self,j,k,fband=0,fwindow=0,dist='gauss',plot=True,err=False,histo=False):
        r"""Plot the instantaneous difference frequency vs normalized bispectral modulus.

        Here we define 

        .. math::

            \Delta f_{\rm inst}(t) \equiv \frac{1}{2\pi}\frac{d\beta(t)}{dt},

        where, for a single value in bifrequency space :math:`(f_1,f_2)`, 
        the local bispectrum :math:`\widetilde{\mathcal{B}}` is given by

        .. math::

            \widetilde{\mathcal{B}}(t) = X(t,f_1)X(t,f_2)X^*(t,f_1+f_2) 
            = |\widetilde{\mathcal{B}}(t)| e^{i \beta(t)},

        where :math:`X(t,f)` is a time-frequency representation. The biphase :math:`\beta(t)` is then

        .. math::

            \beta(t) = \varphi(t,f_1) + \varphi(t,f_2) - \varphi(t,f_1+f_2),

        defining the phases :math:`\varphi(t,f)` via :math:`X(t,f) = |X(t,f)|e^{i\varphi(t,f)}`.

        Args:
            j (int): Index 1.
            k (int): Index 2.
            fband (float): Instantaneous frequency bandwidth. 
                **Presently affects only the output distribution!**
            fwindow (float): Limits for x (frequency) axis.
            dist (str): Choose Gaussian (``'gauss'``) or Lorentzian distribution.
            plot (bool): Plot inst diff freq vs. time.
            err (bool): Plot errorbars instead of timeline.
            histo (bool): Plot phase histogram.

        Returns:
            list: ``freq, amp, freq_err, um = InstDiffFreq(...)``

            * freq (:class:`~numpy.ndarray`) - Instantaneous difference frequency.
            * amp (:class:`~numpy.ndarray`) - Normalized bispectral modulus.
            * freq_err (:class:`~numpy.ndarray`) - Std dev of inst diff freq.
            * um (:class:`~numpy.ndarray`) - Contrived freq-amp distribution.

        """
        b2est,_,Bi = GetBispec(self.sg,self.BicVec,self.LilGuy,j=j,k=k)
        dt = self.tv[1]-self.tv[0]
        dBeta_dt = dphase_dt(Bi) / dt
        d_dt_err = np.sqrt( 2 * (1/b2est - 1) / len(self.tv) ) / dt 
        amp = np.abs(Bi)

        freq = dBeta_dt/(2*np.pi)
        freq_err = d_dt_err / (2*np.pi)
        amp = amp/np.max(amp)

        fband = self.SampRate/200 if fband==0 else fband
        fwindow = self.SampRate/100 if fwindow==0 else fwindow

        # What distribution to use???
        if dist=='gauss':
            um = np.exp(-(freq/fband)**2) * amp
        else: # Lorentzian
            um = (1 + (freq/fband)**2 )**-1 * amp

        ##### Phase histogram
        if histo:
            N = 100
            width = 0.1
            fig,ax = plt.subplots(dpi=self.PlotDPI)
            flim = fwindow/10**self.FScale
            f = np.linspace(-flim,flim,N)
            cnt,_  = np.histogram(freq,
                                  bins=N, range=(-flim,flim),
                                   weights=amp**1,density=not True )
            intcnt = sum(cnt) * ( f[1] - f[0] )

            SaveAs = None
            ax.bar(f,cnt/sum(cnt)*100,width,alpha=1)
            # ax.set_ylim(0,ylim)
            ax.set_xlim(-flim,flim)
            PlotLabels(fig,ax,[r'$\Delta f_{\rm inst}~{\rm [%sHz]}$' % (ScaleToString(self.FScale)), r'${\rm \%}$'],fsize=self.FontSize)
            plt.tight_layout()
            if SaveAs is None:
                plt.show()
            else:
                fig.savefig(SaveAs,dpi=self.PlotDPI,bbox_inches='tight')
                plt.close(fig)
        #######

        if plot:
            fig,ax = plt.subplots(dpi=self.PlotDPI)

            # ax.plot(freq,amp,'o')
            # bic.PlotLabels(plt.gcf(),plt.gca(),[r'$\Delta f_{\rm inst}~{\rm [Hz]}$',r'$|\widetilde{\mathcal{B}}|~{\rm [arb.]}$'],grid=True,fsize=20)

            if err:
                ax.errorbar(freq/10**self.FScale,amp,xerr=freq_err/10**self.FScale,linestyle='',marker='.',markersize=2,capsize=2.0,ecolor='red',mec='C0')
            else:
                PlotTimeline(freq/10**self.FScale,amp,t=self.tv/10**self.TScale,fig=fig,ax=ax,cbar=r'$t\, [\rm{%ss}]$' % (ScaleToString(self.TScale)),cmap=self.TLineCol)
            ax.set_xlim(-fwindow/10**self.FScale,fwindow/10**self.FScale)
            ax.set_ylim(0,1)
            
            PlotLabels(fig,ax,[r'$\Delta f_{\rm inst}~{\rm [%sHz]}$' % (ScaleToString(self.FScale)),r'$|\widetilde{\mathcal{B}}|/|\widetilde{\mathcal{B}}|_{\rm max}$'])
            plt.tight_layout()
            # fig.savefig(specstr,dpi=q.PlotDPI,bbox_inches='tight')
            # plt.close(fig)
            plt.show()

        return freq, amp, freq_err, um


    def InstAmpFreq(self,j,calc_type='hilbert',fband=0,realBPF=True,avPoints=50,IsFreq=False):
        """Perform instantaneous amplitude and frequency analysis.

        .. caution::

            This method is not yet complete... Use at your own peril!

        """
        if IsFreq:    
            _,j = arrmin(abs( self.fv/10**self.FScale - j ))
        
        f0 = self.fv[j]     

        fband = self.SampRate/200 if fband<=0 else fband
        
        Nt = len(self.tv) if calc_type=='spectro' else len(self.Processed)
        
        if calc_type=='spectro':  
            
            t = self.tv/10**self.TScale
            df = self.fv[1]-self.fv[0]
            dt = self.tv[1]-self.tv[0]
            
            dum = self.sg[j,:,0]
            
        elif calc_type in ['hilbert','zerocross']:
            
            dt = 1/self.SampRate
            t = (self.TZero + np.arange(len(self.Processed)) * dt )/10**self.TScale
            df = self.SampRate / len(self.Processed)
            flim = self.fv[j]
            if fband<0:
                dum = self.Processed[:,0]
            else:
                if realBPF:
                    dum = ApplyRealBandpass(self.Processed[:,0],self.SampRate,flim,fband)
                else:
                    dum = ApplyBandpass(self.Processed[:,0],df,flim,fband)
            
            if calc_type=='hilbert':
                dum = hilbert(dum)
            else:
                # Amplitudes are boxcar averages
                amp = boxcar_ave(dum,avPoints)
                # Interpolate for convenience!
                Ninterp = None if self.SampRate/f0 > 10 else int(10*len(dum)*f0/self.SampRate)
                # if Ninterp is not None: 
                #     T,dum = WhittakerShannon(dum,Ninterp,fS=1/dt,T0=self.TZero)
                #     dt=T[1]-T[0]
                T,freq = InstFreqZeroCross(dum,dt=dt,Ninterp=Ninterp,T0=self.TZero)
                freq = np.interp(t,T/10**self.TScale,freq)
                

        elif calc_type=='peak':
            maxLine = 0.
            _,Nf = arrmin( abs( f - maxLine ) )
        
            finst = 0*self.tv
            for k in range(len(self.tv)):
                _,m = arrmin( -abs(self.sg[Nf:,k,self.PlotSig]))
                finst[k] = self.fv[Nf+m]  
       
        if calc_type!='zerocross':
            amp = np.abs(dum)
            freq = dphase_dt(dum) / dt / (2*np.pi)
        return amp, freq, t, dum


    def PlotInstFreq(self,j,k,diff_freq=True,freq_type='hilbert',fband=0,fwindow=0,realBPF=True,SaveStr='',dWin=None):
        """Perform instantaneous frequency analysis.

        .. hint::

            See :func:`BicAn.InstDiffFreq` for more info on the *instantaneous difference frequency*!

        Args:
            j (int): Index 1.
            k (int): Index 2.
            diff_freq (bool): Plot difference frequency instead of individual inst freqs (``False``).
            freq_type (str): Desired means of estimatinf instantaneous frequencies.
                Present support for time derivative of spectrogram (``'spectro'``), 
                Hilbert transform (``'hilbert'``), or zero crossings (``'zerocross'``).
            fband (float): Bandpass filter bandwidth.
            fwindow (float): Limits for x (frequency) axis.
            realBPF (bool): Use Butterworth bandpass (brickwall if ``False``).
            SaveStr (str): Output filename.
            dWin (list of int): Plot "region of interest" from ``dWin[0]`` to ``dWin[1]``. 

        Returns:
            :class:`~numpy.ndarray`: Array of instantaneous frequencies.
        """
        arr = [j,k,j+k]

        fband = self.SampRate/200 if fband==0 else fband
        fwindow = self.SampRate/100 if fwindow==0 else fwindow
        
        b2est,_,Bi = GetBispec(self.sg,self.BicVec,self.LilGuy,j=j,k=k)
        
        Nt = len(self.tv) if freq_type=='spectro' else len(self.Raw)
        X = np.zeros( (3,Nt))
        if freq_type=='zerocross' and diff_freq:
            Y = np.zeros( (3,Nt))
        
        for n in range(3):
        
            if freq_type=='spectro':  
                
                t = self.tv/10**self.TScale
                df = self.fv[1]-self.fv[0]
                dt = self.tv[1]-self.tv[0]
                
                dum = self.sg[arr[n],:,0]
                
            elif freq_type in ['hilbert','zerocross']:
                
                dt = 1/self.SampRate
                t = (self.TZero + np.arange(len(self.Raw)) * dt )/10**self.TScale
                df = self.SampRate / len(self.Raw)
                flim = self.fv[arr[n]]
                if realBPF:
                    dum = ApplyRealBandpass(self.Raw[:,0],self.SampRate,flim,fband)
                else:
                    dum = ApplyBandpass(self.Raw[:,0],df,flim,fband)
                
                if freq_type=='hilbert':
                    dum = hilbert(dum)
                else:
                    Ninterp = None if self.SampRate/flim > 50 else int(50*len(dum)*flim/self.SampRate)
                    T,freq = InstFreqZeroCross(dum,dt=dt,Ninterp=Ninterp,T0=self.TZero)
                    # Interpolate for convenience!
                    freq = np.interp(t,T/10**self.TScale,freq)
       
            if freq_type=='zerocross':
                X[n,:] = 2*np.pi*np.abs( freq )
                # If zerocross, go ahead and calculate hilbert transform so you have all three! 
                if diff_freq: 
                    Y[n,:] = np.abs( np.gradient( np.unwrap(np.angle( hilbert(dum) ))) / dt )

            else:
                X[n,:] = np.abs( np.gradient( np.unwrap(np.angle(dum))) / dt )
                
        
        A3 = np.abs(self.sg[j+k,:,0])
        
        fig,ax = plt.subplots(dpi=self.PlotDPI)   
        
        if not diff_freq:
            # ax.plot( t, X[0,:]/(2*np.pi) - self.fv[j] )
            # ax.plot( t, X[1,:]/(2*np.pi) - self.fv[k] )
            # ax.plot( t, X[2,:]/(2*np.pi) - self.fv[j+k] )

            dumstr = r'$\,\pm\,%d\,\mathrm{%sHz}$' % (fband/10**self.FScale,ScaleToString(self.FScale))

            # Aligns with dissertation convention
            ax.plot( t, (X[1,:]/(2*np.pi) - self.fv[k])/10**self.FScale , label=r'$%d$' % (self.fv[k]/10**self.FScale) + dumstr, lw=2)
            ax.plot( t, (X[0,:]/(2*np.pi) - self.fv[j])/10**self.FScale , label=r'$%d$' % (self.fv[j]/10**self.FScale) + dumstr, lw=2)
            ax.plot( t, (X[2,:]/(2*np.pi) - self.fv[j+k])/10**self.FScale , label=r'$%d$' % (self.fv[j+k]/10**self.FScale) + dumstr, lw=2)
            
            # Attempt at coupling coefficient
            #ax.plot( t, ( Z/(2*np.pi) - self.fv[j+k] ) / (np.imag(Bi) / A3) )
            
            specstr = '%s_InstFreq.png' % SaveStr

            ax.legend(fontsize=2*self.FontSize//3)

        else:
            ax.plot( t, (X[0,:]+X[1,:]-X[2,:])/(2*np.pi) /10**self.FScale, label=r'$1/T$', lw=2)
            
            dBeta_dt = ( np.gradient( np.unwrap(np.angle(Bi))) / ( self.tv[1]-self.tv[0]) ) /10**self.FScale
            ax.plot( self.tv/10**self.TScale, dBeta_dt/(2*np.pi) , label=r'$\dot{\beta}/2\pi$', lw=2)

            if freq_type=='zerocross' and diff_freq:
                ax.plot( t, (Y[0,:]+Y[1,:]-Y[2,:])/(2*np.pi) /10**self.FScale, label=r'$\dot{\phi}(x_a)/2\pi$', lw=2)

            # Uncertainty as given in Fackrell
            # For derivative, we have \sigma_f' = \sqrt(2) * sigma_f / dt
            d_dt_err = np.sqrt( 2 * (1/b2est - 1) / len(self.tv) ) / (self.tv[1]-self.tv[0]) / (2*np.pi) /10**self.FScale
            ax.fill_between( self.tv/10**self.TScale, dBeta_dt/(2*np.pi) - d_dt_err, dBeta_dt/(2*np.pi) + d_dt_err, color='C1', alpha=0.2)
            
            specstr = '%s_DiffFreq.png' % SaveStr
        
        ax.set_xlim(t[0],t[-1])    
        ax.set_ylim(-fwindow/10**self.FScale,fwindow/10**self.FScale)

        if freq_type=='zerocross' and diff_freq:
            ax.legend(fontsize=2*self.FontSize//3)

        tstr = r'$t\, [\mathrm{%ss}]$' % (ScaleToString(self.TScale))
        fstr = r'$\, [\mathrm{%sHz}]$' % (ScaleToString(self.FScale))
        labstr = r'$\delta f$' if not diff_freq else r'$\Delta f_{\rm inst}$'
        # PlotLabels(fig,ax,[tstr,r'$\Delta f_{\rm inst}\,{\rm [Hz]}$'])  
        PlotLabels(fig,ax,[tstr,labstr+fstr])  
        
        if dWin is not None:
            ax.axvspan(dWin[0],dWin[1],color='cyan',alpha=0.2)
        
        plt.tight_layout()
        if len(SaveStr)>0:
            fig.savefig(specstr,dpi=self.PlotDPI,bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        return X


    def PlotPhaseDist(self,j,k,ylim=1,SaveAs=None):
        """Plots phase distribution of single point in bifrequency space.

        Args:
            j (int): Index 1.
            k (int): Index 2.
            ylim (float): Y-axis limit.
            SaveAs (str): Output filename.

        Returns:
            list: ``[<phase bins>, <unweighted pdf>, <weighted pdf>]``
        """
        fig,ax = plt.subplots(dpi=self.PlotDPI)

        N = 16
        width = 0.1
        xlim = [-np.pi,np.pi]
        x = np.linspace(xlim[0],xlim[1],N)
        b2,B,Bi = GetBispec(self.sg,j=j,k=k)
        cnt0,_  = np.histogram(np.angle(Bi),
                              bins=N, range=(xlim[0],xlim[1]),density=not True )
        cntw,_  = np.histogram(np.angle(Bi),
                              bins=N, range=(xlim[0],xlim[1]),
                               weights=np.abs(Bi)**2,density=not True )
        intcnt0 = sum(cnt0) * ( x[1] - x[0] )/np.pi
        intcntw = sum(cntw) * ( x[1] - x[0] )/np.pi

        ax.bar(x/np.pi-width/10,cnt0/sum(cnt0),width,alpha=0.5)
        ax.bar(x/np.pi+width/10,cntw/sum(cntw),width,alpha=0.5)
        ax.set_ylim(0,ylim)
        ax.set_xlim(-1,1)
        PlotLabels(fig,ax,[r'$\beta/\pi$',r'$p(\beta)$'],fsize=self.FontSize,minorgrid=False)
        plt.tight_layout()
        if SaveAs is None:
            plt.show()
        else:
            fig.savefig(SaveAs,dpi=self.PlotDPI,bbox_inches='tight')
            plt.close(fig)

        # print(sum(cnt0),intcnt0,sum(cntw),intcntw)
        return [x,cnt0,cntw]


# Module methods

def FileDialog():
    """Simple file picker dialog.

    Stolen from StackExchange!

    Returns:
        str: User answer.
    """
    root = tk.Tk()
    root.withdraw()
    # Build a list of tuples for each file type the file dialog should display
    my_ftypes = [('all files', '.*'), ('text files', '.txt'), ('dat files', '.dat')]
    # Ask the user to select a single file name
    ans = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select a file:",filetypes=my_ftypes)  
    return ans


def WhittakerShannon(x,Ninterp,fS=1.0,T0=0.0,interp='func'):
    """Whittaker-Shannon interpolation.

    Args:
        x (:class:`~numpy.ndarray`): Data to interpolate.
        Ninterp (int): Number of interpolation points.
        fS (float): Data sampling rate.
        T0 (float): Initial time.
        interp (str): Type of interpolation.
            ``'func'``/``'diff'``/``'int'`` for function/derivative/integral.

    Returns:
        list: ``[<interpolated time>, <interpolated data>]``
    """
    N = len(x)
    Tfinal = N/fS
    T = np.linspace(0,Tfinal,Ninterp)
    D = 0*T
    dsinc = lambda t: t * (t==0) + (t!=0) * (t * np.cos(np.pi*t) - np.sin(np.pi*t)/np.pi ) / t**2
    print('Interpolating %s...' % interp)
    for k in range(N):
        if (k%100)==0:
            LoadBar(k,N)
        # D += fS*x[k]*dsinc(T*fS-k) if diff else x[k]*np.sinc(T*fS-k) 
        if interp=='func':
            D += x[k]*np.sinc(T*fS-k) 
        elif interp=='diff':
            D += fS*x[k]*dsinc(T*fS-k)
        elif interp=='int':
            si,_ = sici(np.pi * (T*fS-k))
            D += x[k]*si / (np.pi*fS)

    print('done!')
    return T+T0,D


def InstFreqZeroCross(x,dt=1.0,crossType='both',Ninterp=None,T0=0.0):
    """Calculates instantaneous frequency from zero crossings.

    Args:
        x (:class:`~numpy.ndarray`): Data to analyze.
        dt (float): Time interval between samples.
        crossType (str): Zero-crossing method.
            Restrict to ``'pos2neg'``, ``'neg2pos'``, or ``'both'``.  
        Ninterp (int): Number of interpolation points.
        T0 (float): Initial time.

    Returns:
        list: ``[<time>, <frequency>]``
    """

    T = T0 + np.arange(len(x))*dt
    if Ninterp is not None:
        T,x = WhittakerShannon(x,Ninterp,fS=1/dt,T0=T0)
        dt=T[1]-T[0]

    if crossType in ['pos2neg','both']:
        pos = x > 0
        p = (pos[:-1] & ~pos[1:]).nonzero()[0]
        fp = (dt * np.diff(p)) ** -1
        loc_pos = (p[1:] + p[:-1])//2
    if crossType in ['neg2pos','both']:
        neg = x < 0
        n = (neg[:-1] & ~neg[1:]).nonzero()[0]
        fn = (dt * np.diff(n)) ** -1
        loc_neg = (n[1:] + n[:-1])//2
    loc = np.concatenate((loc_pos,loc_neg))
    freq = np.concatenate((fp,fn))

    lsort = np.argsort(loc)
    return T[np.sort(loc)], freq[lsort] # np.sort(loc)


def PlotLabels(fig,ax,strings=['x','y'],fsize=20,cbarNorth=False,im=None,cax=None,
    fweight='normal',tickweight='bold',cbarweight='none',grid=True,minorgrid=True,
    shrink=0.7,cbarfsize=None,forceGrid=False,cbarPad=0.05,minorgridColor=[0.9,0.9,0.9],extend='neither'):
    """General purpose plot labels.

    Args:
        fig (:obj:`~matplotlib.figure.Figure`): Figure.
        ax (:obj:`~matplotlib.axes.Axes`): Axes.
        strings (list of str): List of labels for x, y, (z, and colorbar) axes
        fsize (float): Label font size.
        cbarNorth (bool): Place colorbar above plot (to right if ``False``).
        im (:obj:`~matplotlib.collections.QuadMesh`): Image data from, e.g., :obj:`~matplotlib.pyplot.pcolormesh`.
        cax (:obj:`~matplotlib.axes.Axes`): Colorbar axes.
        fweight (str): Font weight
        tickweight (str): Tick label weight.
        cbarweight (str): Colorbar label weight.
        grid (bool): Show grids.
        minorgrid (bool): Show minor grid.
        shrink (float): Colorbar shrink fraction for 3D plots.
        cbarfsize (float): Colorbar font size.
        forceGrid (bool): Force grid.
        cbarPad (float): Colorbar padding fraction.
        minorgridColor (list of float): Color of minor gridlines.
        extend (str): Extend ends of colorbar.

    Returns:
        :obj:`~matplotlib.axes.Axes`: Colorbar axes.
    """
    n = len(strings)

    # Reduce fontsize for 3D plots
    fsize = fsize if n<4 else 3*fsize/4

    # Kill grid if n>2
    grid = grid if (n<=2 or forceGrid) else False

    cbarfsize = 3*fsize/4 if cbarfsize is None else cbarfsize

    # Initialize list for tick label info
    labels = []

    ax.set_xlabel(strings[0], fontsize=fsize, fontweight=fweight)
    if n>1:
        ax.set_ylabel(strings[1], fontsize=fsize, fontweight=fweight)
        # For rotated y axis labels
        # ax.set_ylabel('abc', rotation=0, fontsize=20, labelpad=20)
    if n==3:
        if cax is None:
            divider = make_axes_locatable(ax)
            cbarloc = 'top' if cbarNorth else 'right'
            cax = divider.append_axes(cbarloc, size='5%', pad=cbarPad)
        else:
            cax.clear()
        if cbarNorth:
            fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)   
            cax.xaxis.set_label_position('top') 
            cax.xaxis.tick_top()     
            cax.set_xlabel(strings[2], fontsize=fsize, fontweight=fweight)
        else:
            fig.colorbar(im, cax=cax, extend=extend)
            cax.set_ylabel(strings[2], fontsize=fsize, fontweight=fweight)
        cax.tick_params(labelsize=cbarfsize)
        if cbarweight=='ticks':
            labels += cax.get_xticklabels()
            labels += cax.get_yticklabels()
    if n==4: # Must be trispectrum (or 3D plot)

        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(strings[2], fontsize=fsize, fontweight=fweight, rotation=90)

        if strings[3]!='':
            cbar = fig.colorbar(im,cax=None,ax=ax,shrink=shrink,orientation='horizontal')

            # cbar.ax.set_ylabel(strings[3], fontsize=fsize, fontweight=fweight)
            cbar.ax.set_xlabel(strings[3], fontsize=fsize, fontweight=fweight)
            cbar.ax.tick_params(labelsize=cbarfsize)

        labels += ax.get_zticklabels()
        # labels += cbar.ax.get_xticklabels()
        # labels += cbar.ax.get_yticklabels()

    ax.tick_params(labelsize=9*fsize/10)
    if n<4:
        ax.minorticks_on()

    # Append ticklabels
    labels += ax.get_xticklabels()
    labels += ax.get_yticklabels()    

    # THE MAGIC HAPPENS HERE!
    # This is a robust solution, and doesn't have bad practice "set current axis" calls
    for label in labels:
        label.set_fontweight(tickweight)

    if grid:
        ax.grid(visible=True,which='major')
        if minorgrid:
            ax.grid(visible=True,which='minor',linewidth=0.5,color=minorgridColor)
    return cax


def PlotRHS(x,y,ax,r_col='gray',ylab='',ylim=[],fsize=20,alph=1.0,lw=2):
    """Add plot and right-hand side (RHS) label to existing plot.

    Args:
        x (:class:`~numpy.ndarray`): X data.
        y (:class:`~numpy.ndarray`): Y data.
        ax (:obj:`~matplotlib.axes.Axes`): Axes to plot on.
        r_col (str or list): Data/label color.
        ylab (str): Y axis label.
        ylim (list of float): Y axis limits.
        fsize (float): Font size.
        alph (float): Alpha (transparency) parameter.
        lw (float): Linewidth.

    Returns:
        :obj:`~matplotlib.axes.Axes`: Twinned axes.
    """
    ax_r = ax.twinx()

    ax_r.plot(x, y, color=r_col, alpha=alph, lw=lw)

    labels = []
    labels += ax_r.get_yticklabels()
    for label in labels:
        label.set_fontweight('bold')
        label.set_color(r_col)
    if len(ylim)!=0:
        ax_r.set_ylim(ylim[0],ylim[1])
    ax_r.set_ylabel(ylab,fontsize=fsize,color=r_col)
    ax_r.tick_params(labelsize=9*fsize/10)
    ax_r.minorticks_on()
    return ax_r


def PlotTop(x,y,ax,col='C0',xlab='',xlim=[],fsize=20,alph=1.0,lw=2):
    """Add plot and top label to existing plot.

    Args:
        x (:class:`~numpy.ndarray`): X data.
        y (:class:`~numpy.ndarray`): Y data.
        ax (:obj:`~matplotlib.axes.Axes`): Axes to plot on.
        col (str or list): Data/label color.
        xlab (str): X axis label.
        xlim (list of float): X axis limits.
        fsize (float): Font size.
        alph (float): Alpha (transparency) parameter.
        lw (float): Linewidth.

    Returns:
        :obj:`~matplotlib.axes.Axes`: Twinned axes.
    """
    ax_t = ax.twiny()

    ax_t.plot(x, y, color=col, alpha=alph, lw=lw,)

    labels = []
    labels += ax_t.get_xticklabels()
    for label in labels:
        label.set_fontweight('bold')
        label.set_color(col)
    if len(xlim)!=0:
        ax_t.set_xlim(xlim[0],xlim[1])
    ax_t.set_xlabel(xlab,fontsize=fsize,color=col)
    ax_t.tick_params(labelsize=9*fsize/10)
    ax_t.minorticks_on()
    return ax_t


def SignalGen(fS=1.0,tend=100,Ax=1,fx=1,Afx=0,Ay=0,fy=0,Afy=0,Az=0,Ff=0,noisy=2):
    """3-oscillator FM signal generator.

    Output is defined as sum of three frequency-modulated tones + white noise 

    .. code-block:: text

        sig = Ax*x(t) + Ay*y(t) + Az*z(t) + noisy*noise.

    Note:
        The instantaneous frequency of the third oscillation is tied to the first two!
        That is, ``fz = fx + fy`` and ``dfz = dfx + dfy``.

    Args:
        fS   (float): Sampling frequency in Hz.
        tend (float): End time, via ``t = 0:1/fS:tend``.
        Ax   (float): Amplitude of oscillation #1
        fx   (float): Frequency of oscillation #1 in Hz
        Afx  (float): Amplitude of frequency sweep
        Ay   (float): Amplitude of oscillation #2.
        fy   (float): Frequency of oscillation #2 in Hz.
        Afy  (float): Amplitude of frequency sweep in Hz/s.
        Az   (float): Amplitude of oscillation #3.
        Ff   (float): Frequency of frequency mod.
        noisy (float): Noise amplitude.

    Returns:
        list: ``sig,t,fS = SignalGen(...)``

        * sig (:class:`~numpy.ndarray`) - Test signal.
        * t (:class:`~numpy.ndarray`) - Time vector.
        * fS (:obj:`float`) - Sampling rate.

    """       
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

    sig = x + y + z + noisy*(np.random.random(len(t)) - 0.5)
    #sig = np.reshape(sig, ( len(sig), 1 )) # Output Nx1 numpy array
    return sig,t,float(fS)


def TestSignal(whatsig,tend=100,noisy=2,fS=200,f1=19,f2=45):
    """Provides various signals for bispectral analysis.

    Essentially a suite of test functions for PyBic!

    Mostly a wrapper for :func:`~pybic.SignalGen`

    For more discussion see `RiggsKoepkeMatheny2026`_.

    Args:
        whatsig (str): Input string (see below).
        tend    (float): End time, via ``t = 0:1/fS:tend``.
        noisy   (float): Noise amplitude.
        fS      (float): Sampling frequency in Hz.
        f1      (float): Frequency of oscillation #1 in Hz.
        f2      (float): Frequency of oscillation #2 in Hz.

    Returns:
        list: ``inData,t,fS = TestSignal(...)``

        * inData (:class:`~numpy.ndarray`) - Test signal.
        * t (:class:`~numpy.ndarray`) - Time vector.
        * fS (:obj:`float`) - Sampling rate.

    The following input strings are supported

    ``'demo'``
    ``'classic'``
    ``'tone'``
    ``'noisy'``
    ``'2tone'``
    ``'3tone'``
    ``'4tone'``
    ``'line'``
    ``'circle'``
    ``'fast_circle'``
    ``'quad_couple'``
    ``'d3dtest'``
    ``'cube_couple'``
    ``'coherence'``
    ``'cross_2tone'``
    ``'cross_3tone'``
    ``'cross_circle'``
    ``'amtest'``
    ``'quad_couple_circle'``
    ``'quad_couple_circle2'``
    ``'inst_freq_test'``
    ``'linear_phase'``
    ``'phase_mod'``
    ``'linear_phase_am'``
    ``'phase_mod_am'``
    ``'3tone_short'``
    ``'circle_oversample'``
    ``'cross_3tone_short'``
    ``'helix'``

    Example:
        >>> x,t,fS = TestSignal('quad_couple')
        >>> b = bic.BicAn(x,samprate=fS)

    .. _RiggsKoepkeMatheny2026:
        https://github.com/rigzridge/pybic/blob/main/BicAn_An%20integrated%2C%20open-source%20framework%20for%20polyspectral%20analysis_PREPRINT.pdf

    """ 
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
    elif dum == 'linear_phase':
        t = np.arange(0,tend,1/fS)
        phi = 2 * np.pi * t/tend
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = np.cos(2*np.pi*f2*t + phi) + np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*(f1+f2)*t) + nz/2
    elif dum == 'phase_mod':
        t = np.arange(0,tend,1/fS)
        phi = np.pi * (2*t/tend + (1/2) * np.sin(2*np.pi*t / tend)) # Phase osc. 
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = np.cos(2*np.pi*f2*t + phi) + np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*(f1+f2)*t) + nz/2
    elif dum == 'linear_phase_am':
        t = np.arange(0,tend,1/fS)
        A = (1-0.9*np.sin(np.pi*t/tend)**2)
        phi = 2 * np.pi * t/tend 
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = np.cos(2*np.pi*f2*t + phi) + np.cos(2*np.pi*f1*t) + A * np.cos(2*np.pi*(f1+f2)*t) + nz/2
    elif dum == 'phase_mod_am':
        t = np.arange(0,tend,1/fS)
        A = (1-0.9*np.cos(np.pi*t/tend)**2)
        phi = np.pi * (-1/2 + 2*t/tend + (1/np.pi) * np.sin(2*np.pi*t / tend )) # Phase osc.
        nz,_,_ = SignalGen(fS,tend,Ax=0,fx=0)
        inData = np.cos(2*np.pi*f2*t + phi) + np.cos(2*np.pi*f1*t) + A * np.cos(2*np.pi*(f1+f2)*t) + nz/2
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
        f1 = 108 #97
        f2 = 87  #84
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
        # x,t,_ = SignalGen(fS,tend,fx=24,noisy=0)
        # y,_,_ = SignalGen(fS,tend,fx=37,noisy=0)
        # z,_,_ = SignalGen(fS,tend,fx=41,noisy=0)
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
        t = np.arange(0,tend,1/fS)  # Time-vector sampled at "fS" Hz
                     
        fx = 42.
        fy = 25.
        fz = 0.

        Ff = 0.05
        #Af = 3. / (2*pi*Ff);
        Af = 8. * np.exp(-2*t/tend) / (2*np.pi*Ff);

        dfx = Af * np.sin(2*np.pi*t*Ff)                  # delta f1
        dfy = Af * np.cos(2*np.pi*t*Ff)                  # delta f2
        dfz = 0.5 * t**2 * 2./10
        x = np.sin( 2*np.pi*(fx*t + dfx) )              # f1
        y = np.sin( 2*np.pi*(fy*t + dfy) )              # f2
        z = np.sin( 2*np.pi*(fz*t + dfz) )              # f3

        # phi = 0.5 * np.pi * np.sin(np.pi*t/tend)**2;
        phi = 2*np.pi * t/tend

        u = np.sin( 2*np.pi*(fx*t + fy*t + fz*t + dfx + dfy + dfz) + phi) # f1 + f2

        inData = x + y + z + u + noisy*(0.5*np.random.random(len(t)) - 1)
    elif dum == 'inst_freq_test':
        fS = 500
        t = np.arange(0,1,1/fS)
        Ffin = fS/2
        y = np.sin(2*np.pi * Ffin * t**3 / 3 )

        fx = 100
        Ff = 10
        Afx = 5
        dfx = Afx*np.sin(2*np.pi*t*Ff) / (2*np.pi*Ff)

        y_int = fx + Afx*np.cos(2*np.pi*t*Ff)
        y_int = np.concatenate((y_int,Ffin * t**2))

        y = np.concatenate((np.sin(2*np.pi*(fx*t + dfx)), y))
        t = np.concatenate((t, t+t[-1]))

        inData = y + 0.5*noisy*(np.random.random(len(t)) - 1)
    else:
        print('***WARNING*** :: "{}" test signal unknown... Using single tone..'.format(whatsig)) 
        inData,t,fS = SignalGen(fS,tend,1,22,0,0,0,0,0,0,0)
    return inData,t,float(fS)


def ApplySTFT(sig,samprate=1.0,subint=512,step=256,nfreq=256,t0=0,detrend=False,errlim=1.0e15,window='hann'):
    """Calculate short-time Fourier transform (STFT) of time-series.

    Args:
        sig (:class:`~numpy.ndarray`): Time series to be analyzed.
        samprate (float): Sampling rate in Hz.
        subint (int): Subinterval size in samples.
        step (int): Subinterval step in samples.
        nfreq (int): Number of frequency bins.
        t0 (float): Initial time.
        detrend (bool): Detrend data in each subinterval.
        errlim (float): Max threshold of mean power spectrum.
        window (str): Desired window function.

    Returns:
        list: ``spec,afft,freq_vec,time_vec,err,Ntoss = ApplySTFT(...)``

        * spec (:class:`~numpy.ndarray`) - STFT spectrogram (w/ shape ``(len(sig)/limFreq)``).
        * afft (:class:`~numpy.ndarray`) - Power spectrum.
        * freq_vec (:class:`~numpy.ndarray`) - Frequency vector.
        * time_vec (:class:`~numpy.ndarray`) - Time vector.
        * err (:class:`~numpy.ndarray`) - Mean spectrogram vs time.
        * Ntoss (:obj:`int`) - Number of omitted intervals.

    """
    N = min(sig.shape)
    M = 1 + (max(sig.shape) - subint)//step
    lim  = nfreq                    # Most likey, lim = |_ Nyquist/res _|
    time_vec = np.zeros(M)          # Time vector
    err  = np.zeros((M,N))          # Mean information
    spec = np.zeros((lim,M,N),dtype=complex)      # Spectrogram
    fft_coeffs = np.zeros((lim,N),dtype=complex)  # Coeffs for slice
    afft = np.zeros((lim,N))        # Coeffs for slice
    Ntoss = 0                       # Number of removed slices
    
    # Apply window
    if window in ['flat','flattop','flattopwin']:
        win = FlatTopWindow(subint) 
    elif window in ['none','rect','rectangle']:
        win = HannWindow(subint,q=0) 
    elif window in ['sin','sine']:
        win = HannWindow(subint,q=1) 
    else: # Must be Hann!
        if window!='hann':
            print('***WARNING*** :: Defaulting to Hann window!')
        win = HannWindow(subint,q=2)        
    
    print('Applying STFT...      ')
    for m in range(M):
        LoadBar(m,M)

        time_vec[m] = t0 + (m*step+subint//2)/samprate
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
                # Keep blank if mean excessive
                Ntoss += 1
            else:
                afft[:,k]  += dumft              # Welch's PSD
                spec[:,m,k] = fft_coeffs[:,k]    # Build spectrogram
    
    freq_vec = np.arange(lim)*samprate/subint
    #freq_vec = freq_vec[0:lim] 
    afft /= M     
    return spec,afft,freq_vec,time_vec,err,Ntoss


def CalcHistVsT(sig,samprate=1.,subint=512,step=256,t0=0,binMax=1.,Nbins=200):
    """Calculate amplitude histogram vs. time of time-series.

    Args:
        sig (:class:`~numpy.ndarray`): Time series to be analyzed.
        samprate (float): Sampling rate in Hz.
        subint (int): Subinterval size in samples.
        step (int): Subinterval step in samples.
        nfreq (int): Number of frequency bins.
        t0 (float): Initial time.
        binMax (float): Max binned amplitude.
        Nbins (int): Number of amplitude bins.

    Returns:
        list: ``hist,mh,binvec,time_vec = CalcHistVsT(...)``

        * hist (:class:`~numpy.ndarray`) - Amplitude histogram vs. time.
        * mh (:class:`~numpy.ndarray`) - Average of histogram over time.
        * binvec (:class:`~numpy.ndarray`) - Vector of bin locations (amplitudes).
        * time_vec (:class:`~numpy.ndarray`) - Time vector.

    """
    N = min(sig.shape)
    M = 1 + (max(sig.shape) - subint)//step
    time_vec = np.zeros(M)          # Time vector
    binvec = np.linspace(-binMax,binMax,Nbins)

    hist = np.zeros((Nbins,M,N))      # Histogram array

    for m in range(M):
        LoadBar(m,M)

        time_vec[m] = t0 + (m*step+subint//2)/samprate
        for k in range(N):
            Ym = sig[m*step : m*step + subint, k] # Select subinterval    

            counts,_ = np.histogram(Ym,bins=Nbins,range=(-binMax,binMax))
            hist[:,m,k] = counts / np.sum(counts)
    
    mh = np.mean(hist,axis=1)   
    return hist,mh,binvec,time_vec


def ApplyCWT(sig,samprate=1.0,sigma=3.14,limFreq=2,alphaExp=0.5):
    """Calculate continuous wavelet transform (CWT) of time-series.

    Here we define 

    .. code-block:: text

        CWT[j,k] = A_sigma k^{alphaExp-0.5} sum_{p=0}^{N-1} X[p] 

                    * exp[ - 2pi^2 (sigma^2 / k^2) (p-k)^2 + 2pi i j p / N  ]  

        with A_sigma = pi^{1/4} sqrt{2N sigma / samprate} and X is the FFT of x[n].

    Args:
        sig (:class:`~numpy.ndarray`): Time series to be analyzed.
        samprate (float): Sampling rate.
        sigma (float): Time-frequency adjustment.
        limFreq (int): Frequency limit division.
        alphaExp (float): Alpha exponent.

    Returns:
        list: ``CWT,acwt,freq_vec,time_vec = ApplyCWT(...)``

        * CWT (:class:`~numpy.ndarray`) - CWT (w/ shape ``(len(sig)/limFreq, len(sig)/2, Nseries)``).
        * acwt (:class:`~numpy.ndarray`) - Power spectrum (w/ length ``len(sig)/limFreq``).
        * freq_vec (:class:`~numpy.ndarray`) - Frequency vector (w/ length ``len(sig)/limFreq``).
        * time_vec (:class:`~numpy.ndarray`) - Time vector (w/ length ``len(sig)/2``).

    """
    Nsig,N = sig.shape
    nyq    = Nsig//2

    lim = int(Nsig/limFreq)

    f0 = samprate/Nsig
    freq_vec = f0 * np.arange(nyq) # Frequency vector as calculated by FFT
    
    acwt = np.zeros((lim,N))
    CWT  = np.zeros((lim,nyq,N),dtype=complex)

    # Morlet wavelet in frequency space
    ###Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma) * a**(alphaExp-0.5) * np.exp( -2 * np.pi**2 * sigma**2 * ( freq_vec/a - f0)**2 )
    ##Psi = lambda a: (np.pi**0.25)*np.sqrt(2) * a**(alphaExp-0.5) * np.exp( -2 * np.pi**2 * (sigma/a)**2 * ( np.arange(nyq) - a)**2 )
    Psi = lambda a: (np.pi**0.25)*np.sqrt(2*sigma/f0) * a**(alphaExp-0.5) * np.exp( -2 * np.pi**2 * (sigma/a)**2 * ( np.arange(nyq) - a)**2 )

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

    time_vec = 2 * np.arange(nyq)/samprate
    return CWT,acwt,freq_vec[0:lim],time_vec


def SpecToCoherence(spec,lilguy=1e-6):
    """Estimate cross-spectrum/-coherence from spectrogram.

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        lilguy (float): Small value to avoid ``0/0`` errors.

    Returns:
        list: ``C,cc,cx = SpecToCoherence(...)``

        * C (:class:`~numpy.ndarray`) - Cross-spectrum.
        * cc (:class:`~numpy.ndarray`) - Cross-coherence spectrum.
        * xx (:class:`~numpy.ndarray`) - "Coherogram".

    """
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
    """Estimate **auto**-bispectrum/-bicoherence spectrum from spectrogram(s).

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        v (list of int): List of time-series' indices to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.

    Returns:
        list: ``b2,B = SpecToBispec(...)``

        * b2 (:class:`~numpy.ndarray`) - Bicoherence spectrum.
        * B (:class:`~numpy.ndarray`) - Bispectrum.

    """
    nfreq,slices,_ = spec.shape

    lim = nfreq

    B  = np.zeros((lim//2,lim),dtype=complex)
    b2 = np.zeros((lim//2,lim))
    
    print('Calculating bicoherence...      ')     
    for j in range(lim//2):
        LoadBar(j,lim//2)
        
        for k in np.arange(j,lim-j):
            # p1 = spec[k,:,v[0]]
            # p2 = spec[j,:,v[1]]
            # s  = spec[j+k,:,v[2]]

            # Bi  = p1 * p2 * np.conj(s)
            # e12 = abs(p1*p2)**2
            # e3  = abs(s)**2

            # Bjk = sum(Bi)                
            # E12 = sum(e12)             
            # E3  = sum(e3)                      

            # b2[j,k] = (abs(Bjk)**2)/(E12*E3+lilguy) 

            # B[j,k] = Bjk

            b2[j,k] , B[j,k] , _ = GetBispec(spec,v=v,lilguy=lilguy,j=j,k=k)

    B = B/slices   
    return b2,B

def SpecToCrossBispec(spec,v=[0,0,0],lilguy=1e-6):
    """Estimate **cross**-bispectrum/-bicoherence spectrum from spectrogram(s).

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        v (list of int): List of time-series' indices to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.

    Returns:
        list: ``b2,B = SpecToCrossBispec(...)``

        * b2 (:class:`~numpy.ndarray`) - Cross-bicoherence spectrum.
        * B (:class:`~numpy.ndarray`) - Cross-bispectrum.

    """
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
    return b2,B


def SpecToTrispec(spec,v=[0,0,0],lilguy=1e-6):
    """Estimate **auto**-trispectrum/-tricoherence spectrum from spectrogram(s).

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        v (list of int): List of time-series' indices to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.

    Returns:
        list: ``t2,T = SpecToTrispec(...)``

        * t2 (:class:`~numpy.ndarray`) - Tricoherence spectrum.
        * T (:class:`~numpy.ndarray`) - Trispectrum.

    """
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

    return t2,T              


def GetBispec(spec,v=[0,0,0],lilguy=1e-6,j=0,k=0,rando=False):
    """Estimate **local** bispectrum and bicoherence of a single (f1,f2) value.

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        v (list of int): List of time-series' indices to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.
        j (int): Index 1.
        k (int): Index 2.
        rando (float): Randomization level (``1.0`` if ``True``).

    Returns:
        list: ``b2,B,Bi = GetBispec(...)``

        * b2 (:obj:`float`) - Bicoherence value.
        * B (:obj:`complex`) - Bispectrum value.
        * Bi (:class:`~numpy.ndarray`) - **Local** (i.e., time-dependent) bispectrum.

    """
    #p1 = spec[k,:,v[0]]
    #p2 = spec[j,:,v[1]]
    #s  = spec[j+k,:,v[2]]

    p1 = np.real( spec[abs(k),:,v[0]] ) + 1j*np.sign(k)*np.imag( spec[abs(k),:,v[0]] )
    p2 = np.real( spec[abs(j),:,v[1]] ) + 1j*np.sign(j)*np.imag( spec[abs(j),:,v[1]] )
    s  = np.real( spec[abs(j+k),:,v[2]] ) + 1j*np.sign(j+k)*np.imag( spec[abs(j+k),:,v[2]] )

    if rando>0:
        # Old way ~> which is redundant anyway, right? using uniform [-1,1] * 2pi?
        # p1 = abs(p1)*np.exp( 2j*np.pi* (2*np.random.random( p1.shape ) - 1) )
        # p2 = abs(p2)*np.exp( 2j*np.pi* (2*np.random.random( p2.shape ) - 1) )
        # s  = abs(s)* np.exp( 2j*np.pi* (2*np.random.random( s.shape  ) - 1) )

        w = 1.0 if isinstance(rando,bool) else rando
        p1 = abs(p1)*np.exp( 1j* ( np.angle(p1) + w*2*np.pi*np.random.random( p1.shape )  ) )
        p2 = abs(p2)*np.exp( 1j* ( np.angle(p2) + w*2*np.pi*np.random.random( p2.shape )  ) )
        s  = abs(s)* np.exp( 1j* ( np.angle(s) + w*2*np.pi*np.random.random( s.shape  )  ) )

    Bi  = p1*p2*np.conj(s)
    e12 = abs(p1*p2)**2
    e3  = abs(s)**2

    B   = sum(Bi)                 
    E12 = sum(e12)            
    E3  = sum(e3)                      

    b2 = (abs(B)**2)/(E12*E3+lilguy)
    
    B = B/len(Bi)
    return b2,B,Bi 

def GetBispecBootstrap(spec,v=[0,0,0],lilguy=1e-6,j=0,k=0,Ntrials=100):
    """Estimate mean and std dev of bicoherence for single (f1,f2) value.

    Uses bootstrapping, i.e., resampling w/ replacement.

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        v (list of int): List of time-series' indices to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.
        j (int): Index 1.
        k (int): Index 2.
        Ntrials (int): Number of random trials.

    Returns:
        list: ``b2boot,Bboot,Bi_meas = GetBispecBootstrap(...)``

        * b2boot (:obj:`float`) - Bootstrap bicoherence value.
        * Bboot (:obj:`complex`) - Bootstrap bispectrum value.
        * Bi_meas (:obj:`complex`) - Estimated bispectrum value.

    """
    p1 = np.real( spec[abs(k),:,v[0]] ) + 1j*np.sign(k)*np.imag( spec[abs(k),:,v[0]] )
    p2 = np.real( spec[abs(j),:,v[1]] ) + 1j*np.sign(j)*np.imag( spec[abs(j),:,v[1]] )
    s  = np.real( spec[abs(j+k),:,v[2]] ) + 1j*np.sign(j+k)*np.imag( spec[abs(j+k),:,v[2]] )

    N = len(s)
    Bboot = np.zeros(Ntrials,dtype=complex)
    b2boot = np.zeros(Ntrials)
    Bi_meas = p1*p2*np.conj(s)

    for k in range(Ntrials):

        # Resample w/ replacement
        r = (np.random.random(N) * N).astype(int)
        dum1 = p1[r]
        dum2 = p2[r]
        dums = s[r]

        Bi  = dum1*dum2*np.conj(dums)
        e12 = abs(dum1*dum2)**2
        e3  = abs(dums)**2

        B   = sum(Bi)                 
        E12 = sum(e12)            
        E3  = sum(e3)                      

        b2boot[k] = (abs(B)**2)/(E12*E3+lilguy)
        Bboot[k] = B/len(Bi)
    
    return b2boot,Bboot,Bi_meas


def GetPolySpec(spec,f,lilguy=1e-6,rando=False,v=[]):
    """Estimate the nth-order polyspectrum/polycoherence spectrum of a given (f1,f2,...,fn) value

    Args:
        spec (:class:`~numpy.ndarray`): Input spectrogram.
        f (list of int): List of frequencies to analyze.
        lilguy (float): Small value to avoid ``0/0`` errors.
        rando (float): Randomization level (``1.0`` if ``True``).
        v (list): List of time-series' indices.

    Returns:
        list: ``nCoh,nSpec,nSpec_i*s = GetPolySpec(...)``

        * nCoh (:obj:`float`) - Polycoherence value.
        * nSpec (:obj:`complex`) - Polyspectrum value.
        * nSpec_i (:class:`~numpy.ndarray`) - **Local** (i.e., time-dependent) polyspectrum.

    """

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
    """Hann window (and all powers of sine).

    Args:
        N (int): Number of samples.
        q (float): Sine window exponent.

    Returns:
        :class:`~numpy.ndarray`: Window data.
    """
    return (np.sin(np.pi*np.arange(N)/(N-1)))**q
    

def FlatTopWindow(N):
    """Flat-top window.

    Args:
        N (int): Number of samples.

    Returns:
        :class:`~numpy.ndarray`: Window data.
    """
    s = 2*np.pi*np.arange(N)/(N-1)
    return 0.22 - 0.42*np.cos(s) + 0.28*np.cos(2*s) - 0.08*np.cos(3*s) + 0.01*np.cos(4*s)


def ApplyBandpass(x,fS=1.0,f0=0.25,fband=0.1):
    """Quick and dirty brickwall bandpass filter.

    Args:
        x (:class:`~numpy.ndarray`): Input time-series.
        fS (float): Sampling frequency in Hz.
        f0 (float): Center frequency (peak of passband).
        fband (float): Filter bandwidth.

    Returns:
        :class:`~numpy.ndarray`: Filtered time-series.
    """

    # This is kind of crazy! But without this things might break...
    x = np.reshape(x,[x.shape[0]])

    N = len(x)
    fftx = np.fft.fft(x)

    Klo  = np.ceil( (N/fS) * (f0 - fband ) ).astype(int)
    Khi  = np.ceil( (N/fS) * (f0 + fband ) ).astype(int)

    # fftx[0:Klo] = 0
    # fftx[(-Klo):-1] = 0
    # fftx[Khi:(-Khi)] = 0
    fftx[0:Klo] = 0
    fftx[(-1-Klo+1):-1] = 0
    fftx[Khi:(-1-Khi+1)] = 0

    return np.real(np.fft.ifft(fftx))    # Invert it!


def ApplyRealBandpass(D,fS,flim,fband,order=5):
    """Butterworth bandpass filter.

    Args:
        D (:class:`~numpy.ndarray`): Input time-series.
        fS (float): Sampling rate in Hz.
        flim (float): Center frequency (peak of passband).
        fband (float): Filter bandwidth.
        order (int): Butterworth filter order.

    Returns:
        :class:`~numpy.ndarray`: Filtered time-series.
    """
    lo = flim - fband
    hi = flim + fband
    
    sos = butter(order, [lo, hi], fs=fS, btype='band', output='sos')
    
    #w,h = sosfreqz(sos,worN=256)
    #plt.plot(w,abs(h))
    #plt.show()
    return sosfiltfilt(sos, D) 


def ApplyDetrend(y):
    """Remove linear trend from data.

    Args:
        y (:class:`~numpy.ndarray`): Input data.

    Returns:
        :class:`~numpy.ndarray`: Detrended data.
    """
    n = len(y)
    dumx  = np.arange(1,n+1) 
    s = (6/(n*(n**2-1))) * (2*sum(dumx*y) - sum(y)*(n+1))
    y = y - s*dumx
    return y


def ScaleToString(scale):
    r"""Converts order of magnitude to `metric prefix`_.

    Args:
        scale (int): Order of magnitude. Support for :math:`\in [-15,15]`.

    Returns:
        str: Metric prefix.

    Example:
        >>> ScaleToString(3)
        'k'
        >>> ScaleToString(-3)
        'm'

    .. _metric prefix:
        https://en.wikipedia.org/wiki/Metric_prefix

    """
    tags = ['f',[],[],'p',[],[],'n',[],[],r'$\mu$',[],[],'m','c','d','', [],'h','k',[],[],'M',[],[],'G',[],[],'T',[],[],'P',[],[],'E']
    s = tags[15+scale]
    return s   


def LoadBar(m, M, bar_length=40):
    """Loading bar animation.

    Args:
        m (int): Current step.
        M (int): Total steps.
        bar_length (int): Length of loading bar.

    .. caution::

        Printing to the shell with ``sys.stdout.write()`` sometimes 
        takes **forever** when using IDLE... *Need to fix this?*

    """
    ch1 = r'||\-/|||'
    ch2 = r'_.:"^":.'
    fraction = (m+1) / M
    arrow = int(fraction * bar_length - 1) * '~' + '>'
    padding = (bar_length - len(arrow)) * ' '
    
    xtra = '^]' if (m+1)==M else ch1[m%8] + ch2[m%8]
    ending = '\n' if (m+1)==M else '\r'
    
    # Print the bar to stderr or stdout
    sys.stdout.write(f'Progress: [{arrow}{padding}] {int(fraction*100)}%{xtra}{ending}')
    # sys.stdout.flush()


def PlotTimeline(x,y,t=None,strings=None,fig=None,ax=None,lw=2,cmap='turbo',cbar=None,
    fsize=14,fweight='normal',xlim=None,ylim=None,cbarNorth=True,forceGrid=True,dpi=180):
    """Plot line with color.

    Sourced from https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html.

    Args:
        x (:class:`~numpy.ndarray`): X data.
        y (:class:`~numpy.ndarray`): Y data.
        t (:class:`~numpy.ndarray`): "time" data.
        fig (:obj:`~matplotlib.figure.Figure`): Input figure.
        ax (:obj:`~matplotlib.axes.Axes`): Input axes.
        lw (float): Linewidth.
        cmap (str): Colormap.
        cbar (bool): Include colorbar.
        fsize (float): Font size.

    """
    from matplotlib.collections import LineCollection

    plotNow = False
    if fig is None:
        plotNow = True
        fig, ax = plt.subplots(dpi=dpi)

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

    strings = ['x','y','t'] if strings is None else strings
    cax = PlotLabels(fig,ax,im=line,strings=strings,fsize=fsize,fweight=fweight,cbarNorth=cbarNorth,forceGrid=forceGrid)

    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        ax.set_xlim(np.min(x),np.max(x))
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ax.set_ylim(np.min(y),np.max(y))

    if plotNow:
        plt.tight_layout()
        plt.show()


def Plot(dats,strings=None,color=None,alpha=1.,marker=None,ms=6,lw=2,ls='-',fsize=20,fweight='normal',grid=True,
                minorgrid=True,minorgridColor=[0.9,0.9,0.9],tickweight='bold',xlim=None,ylim=None,forceGrid=False,
                cmap='CMRmap',cbarNorth=False,cbarweight='none',cbarfsize=None,cbarPad=0.05,vlim=None,cax=None,
                zlim=None,shrink=0.7,elev=26,azim=-30,roll=0,zoom=0.8,dpi=180,SaveAs=None,figax=None):
    """All purpose plotting tool!

    .. hint::

        See the dedicated `Jupyter notebook`_ for more info!

    Args:
        dats (list): Data.

    Returns:
        list: ``fig,ax,cax = Plot(...)``

        * fig (:obj:`~matplotlib.figure.Figure`) - Output figure.
        * ax (:obj:`~matplotlib.axes.Axes`) - Output axes.
        * cax (:obj:`~matplotlib.axes.Axes`) - Colorbar axes (else :obj:`None`).

    .. _Jupyter notebook:
        https://colab.research.google.com/drive/1NJmjnkhD9wWd_uYRYDWSOEatzS_5Nzm3?usp=sharing

    """

    if figax is not None:
        fig = figax[0]
        ax = figax[1]

    # Standard x y plot
    if len(dats)==2:
        if figax is None:
            fig,ax = plt.subplots(dpi=dpi)
        if color is None:
            ax.plot(dats[0],dats[1],lw=lw,linestyle=ls,marker=marker,ms=ms,alpha=alpha)
        else:
            ax.plot(dats[0],dats[1],lw=lw,linestyle=ls,marker=marker,ms=ms,alpha=alpha,color=color)
        if strings is None:
            strings=['x','y']
        cax = PlotLabels(fig,ax,strings=strings,fsize=fsize,cbarNorth=cbarNorth,fweight=fweight,
            tickweight=tickweight,grid=grid,minorgrid=minorgrid,minorgridColor=minorgridColor)

    # Either trajectory in 3D or image/surface plot
    if len(dats)==3:

        # Check image
        if len(dats[2].shape)==2:
            if figax is None:
                fig,ax = plt.subplots(dpi=dpi)
            if vlim is None:
                # Could sub out pcolormesh for e.g., contour/contourf
                im = ax.pcolormesh(dats[0],dats[1],dats[2],cmap=cmap)
            else:
                im = ax.pcolormesh(dats[0],dats[1],dats[2],cmap=cmap,vmin=vlim[0],vmax=vlim[1])
            if strings is None:
                strings=['x','y','c']
            cax = PlotLabels(fig,ax,im=im,cax=cax,strings=strings,fsize=fsize,cbarNorth=cbarNorth,fweight=fweight,forceGrid=forceGrid,cbarPad=cbarPad,
                tickweight=tickweight,cbarweight=cbarweight,cbarfsize=cbarfsize,grid=grid,minorgrid=minorgrid,minorgridColor=minorgridColor)

        # Must be trajectory
        else:
            if figax is None:
                fig = plt.figure(dpi=dpi)
                ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim, roll=roll)
            if color is None:
                ax.plot(dats[0],dats[1],dats[2],lw=lw,linestyle=ls,marker=marker,ms=ms,alpha=alpha)
            else:
                ax.plot(dats[0],dats[1],dats[2],lw=lw,linestyle=ls,marker=marker,ms=ms,alpha=alpha,color=color)
            if strings is None:
                strings=['x','y','z']
            cax = PlotLabels(fig,ax,strings=strings + [''],fsize=fsize,cbarNorth=cbarNorth,fweight=fweight,
                tickweight=tickweight,grid=grid,minorgrid=minorgrid,minorgridColor=minorgridColor)
            ax.set_box_aspect(None, zoom=zoom)

    #  Scalar data from volume
    if len(dats)==4:
        if figax is None:
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

        # Make meshgrid and flatten data
        X, Y, Z = np.meshgrid(dats[0], dats[1], dats[2])
        im = ax.scatter(X.flatten(),Y.flatten(),Z.flatten(),c=dats[3].flatten(),cmap=cmap,alpha=alpha,marker=marker)
        ax.view_init(elev=elev, azim=azim, roll=roll)
        if strings is None:
            strings=['x','y','z','c']
        PlotLabels(fig,ax,im=im,cax=cax,strings=strings,fsize=fsize,cbarNorth=cbarNorth,fweight=fweight,forceGrid=forceGrid,cbarPad=cbarPad,shrink=shrink,
            tickweight=tickweight,cbarweight=cbarweight,cbarfsize=cbarfsize,grid=grid,minorgrid=minorgrid,minorgridColor=minorgridColor)
        # ax.set_box_aspect(None, zoom=zoom)
      
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])

    plt.tight_layout()

    # Check if overplotting
    if figax is None:
        if SaveAs is None:
            plt.show()
    # Always save if user asks
    if SaveAs is not None:
        fig.savefig(SaveAs,dpi=dpi,bbox_inches='tight')
        plt.close(fig)

    return [fig,ax,cax]

def RunDemo():
    """Demonstration of :class:`BicAn`."""
    b = BicAn('circle')
    return b

def arrmin(arr):
    """Matlab-esque ``min()``.
    
    Lol pretty much :func:`~numpy.argmin`.

    Args: 
        arr (:class:`~numpy.ndarray`): Array in question.

    Returns: 
        list: ``[<min value>, <min index>]``
    """  
    m = min(arr)
    index = arr.tolist().index( m )
    return m,index

def dphase_dt(z):
    """Returns gradient of unwrapped phase.

    Args: 
        z (:class:`~numpy.ndarray`): Input data.

    Returns: 
        :class:`~numpy.ndarray`: dphase/dt.
    """
    return np.gradient( np.unwrap(np.angle(z)))

def boxcar_ave(x,N):
    """Smooths data, attempts to return correct amplitudes.

    Args: 
        x (:class:`~numpy.ndarray`): Input data.
        N (int): Number of samples to smooth.

    Returns:
        :class:`~numpy.ndarray`: Smoothed data.
    """
    a = uniform_filter1d( abs(x), size=N )
    # Try to get back to actual amplitudes!
    return a * np.max(abs(x)) / max(a)

def bin_mat(n):
    """Creates array of all combinations of ``[+/-1, +/-1, ...]``.

    Args: 
        n (int): Number of indices.

    Returns:
        :class:`~numpy.ndarray`: Output array.

    Example:
        >>> bin_mat(2)
        array([[ 1, -1],
               [ 1,  1]])
        >>> bin_mat(4)
        array([[ 1, -1, -1, -1],
               [ 1, -1, -1,  1],
               [ 1, -1,  1, -1],
               [ 1, -1,  1,  1],
               [ 1,  1, -1, -1],
               [ 1,  1, -1,  1],
               [ 1,  1,  1, -1],
               [ 1,  1,  1,  1]])

    """
    dum = ((np.arange(2**(n-1),2**n).reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)[:,::-1]
    dum[dum==0] = -1
    return dum

def nRandSumLessThanUnity(n):
    """Outputs n numbers whose sum is < 1.

    Args: 
        n (int): Number of elements in sum.

    Returns:
        list: List of :obj:`float` with sum less than unity.
    """
    foundIt = False
    while not foundIt:
        
        dum = np.random.random(n)

        if sum(dum)<=1:
            foundIt = True
            dum.sort()
            dum = dum[::-1]
            return dum

def DrawSimplex(flim):
    """Draws simplex for trispectrum.

    Args: 
        flim (float): Frequency limit.
    """
    plt.plot([0,flim/3],     [0,flim/3],     [0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim,flim/3],  [0,flim/3],     [0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,flim/3],[flim/2,flim/3],[0,flim/3],color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,0],     [flim/2,0],     [0,0],     color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([flim/2,flim],  [flim/2,0],     [0,0],     color=[0.5,0.5,0.5], lw=2.5)
    plt.plot([0,flim],       [0,0],          [0,0],     color=[0.5,0.5,0.5], lw=2.5)


if __name__ == '__main__':
    b = RunDemo()