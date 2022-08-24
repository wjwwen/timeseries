import datetime
import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%
# Generate some fake data
t_max = 4 # signal duration (seconds)
sample_freq = 250 # points per second
N = t_max*sample_freq
t = np.linspace(0, t_max, N)
amp = np.array([1, .3, .1])
freq = np.array([1, 2, 10])

# Plot the individual components and the total signal
total = np.zeros(N)
components = []

n_freq = len(freq)

for i in range(n_freq):
    current = amp[i]*np.cos(2*np.pi*freq[i]*t)
    total += current
    
    components.append(current)
    plt.plot(t, current, label='f='+str(freq[i]), lw=1)
    
plt.plot(t, total, label='total')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %%
# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("\n\nTime (s)", fontsize=18)
ax.set_ylabel("\n\nFrequency (Hz)", fontsize=18)
ax.set_zlabel("\n\nAmplitude", fontsize=18)

linewidth = 1

n_freq = np.max(freq)+2
x = np.linspace(0,4,1000)
y = np.ones(x.size)

# Plot the total signal
ax.plot(x, y*n_freq, total, linewidth=3, color=colors[1])

# Plot the amplitudes
z = np.zeros(n_freq*100)
z[freq*100] = amp

ax.plot(np.zeros(n_freq*100), np.linspace(0, n_freq, n_freq*100), z, 
        linewidth=3, color=colors[3])

# Plot the components
y = np.ones(1000)
for i in range(0, len(components)):
    ax.plot(x, y*freq[i], components[i], linewidth=1.5, color=colors[0])
    
ax.set_yticks(freq)
ax.set_yticklabels(freq)

ax.set_xlim(0, t_max)
ax.set_ylim(0, n_freq)

# %% 
# Recover the original frequencies and amplitudes from the total signal 
# by taking the fourier transform
fft_values = scipy.fftpack.fft(total)

fft_values.dtype # Array of complex nos.

# To recover the real component of the signal, we simply take the absolute value. The imaginary components corresponds to phase information
fft_real = 2.0/N * np.abs(fft_values[0:N//2])

# We see that only a few values are significantly different from zero:
np.where(fft_real>0.01)

# To properly recover the corresponding freqency values, we must calculate the freqency resolution
freq_resolution = sample_freq/N

# This is the value we need to convert indices into frequencies
freq_values = np.arange(N)*freq_resolution

plt.plot(freq_values[:50], fft_real[:50], label='calculated')
plt.scatter(freq, amp, s=100, color=colors[1], zorder=3, label='original')
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# %%
# Load Dataset
ILI = pd.read_csv('CDC.csv')
ILI['date'] = ILI['Year']+ILI['Week']/52.

# Visualize
ILI.plot(x='date', y=['Percent of Deaths Due to Pneumonia and Influenza', 'Expected', 'Threshold'])
ax = plt.gca()
ax.legend(['Mortality', 'Expected', 'Threshold'])
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')

# %%
# Calculate FFT
signal = ILI['Percent of Deaths Due to Pneumonia and Influenza']
date = ILI['date']
N = len(signal)
fft_values = scipy.fftpack.fft(signal.values)

# Frequencies
freq_values = scipy.fftpack.fftfreq(N, 1/52) # 52 weeks per year

# Plot amplitude as a function of frequency
fig, ax = plt.subplots(1)
ax.semilogy(freq_values[:N//2], np.abs(fft_values[:N//2]))
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('Amplitude')

# Remove some noise by filtering out higher frequency
filtered = fft_values.copy()
filtered[np.abs(freq_values) > 2] = 0

# Reconstructing filtered data
signal_filtered = np.real(sp.fftpack.ifft(filtered))

# Cleaner version of signal
fig, ax = plt.subplots(1)
ax.plot(date, signal, lw=1, label='original')
ax.plot(date, signal_filtered, label=r'$f_{max}=2$')
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')
fig.legend()

# %%
# Naturally, the more frequencies we include the closer we get to the original dataset
filtered2 = fft_values.copy()
filtered2[np.abs(freq_values) > 4] = 0
signal_filtered2 = np.real(sp.fftpack.ifft(filtered2))

fig, ax = plt.subplots(1)
ax.plot(date, signal, lw=1, label='original')
ax.plot(date, signal_filtered, lw=1, label=r'$f_{max}=2$')
ax.plot(date, signal_filtered2, label=r'$f_{max}=4$')
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')
plt.legend()