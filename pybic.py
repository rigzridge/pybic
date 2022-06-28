# Quick attempt at PyBic
# 6/24/2022

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Sampling rate
fS = 1000
dt = 1/fS

# Time base
tmax = 10
t = np.arange(0, tmax, dt)
N = len(t)

# Frequency domain
df = 1/(dt*N)
f = np.arange(0, fS, df) 

# Frequency
w = 10;

# Signal
sig = N*[0]
#sig = np.sin(w*2*np.pi*t) + np.sin(2*w*2*np.pi*t)/2 + np.sin(3*w*2*np.pi*t)/3
for k in range(10):
    sig += np.sin( (k+1) * w * 2*np.pi * t )/(k+1)

# FFT
ft = np.fft.fft(sig)/N

fig  = plt.figure()
ax   = fig.add_subplot(111)

# Plot
plt.plot(f,np.abs(ft))
plt.yscale("log")

ax.set_xlim(0, fS/2)
ax.set_ylim(0.0001, 1)

plt.xlabel("Frequency [Hz]")
plt.ylabel("|P|")

plt.show()
