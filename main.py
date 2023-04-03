#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plotter
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import simpleaudio as sa
import os

import ICA
import plotting_functions as plotting


curr_path = os.path.dirname(os.path.realpath(__file__))

# audio file names:

file_x1=os.path.join(curr_path,'Female Voice.wav')
file_x2=os.path.join(curr_path,'Male Voice.wav')
mixed=os.path.join(curr_path,"mix.wav")
file_s1 = os.path.join(curr_path,"1st voice.wav")
file_s2 = os.path.join(curr_path,"2nd voice.wav")
# plot titles:
title_x = "Observed Data x"
title_s = "Estimated Source s"
title_X = "Observed Data X"
title_S = "Estimated Sources S"

# """
# play audio files:
print("Playing", file_x1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_x2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x2)
play_obj = wave_obj.play()
play_obj.wait_done()
# """

# load audio files:
sample_freq_1, x1 = read(file_x1)
sample_freq_2, x2 = read(file_x2)

v=min(len(x1),len(x2))
S=np.r_[[x1[0:v]],[x2[0:v]]]
A= np.array([[4.5359,-5.6223],[-6.0723, -9.1858]])
X=np.dot(A,S) 
num_sig = X.shape[0]

write(mixed,sample_freq_1,X[0].astype(np.int16))

# play mixed audio files:
print("Playing", mixed, "...")
wave_obj = sa.WaveObject.from_wave_file(mixed)
play_obj = wave_obj.play()
play_obj.wait_done()

# """

# plot raw audio signals:
plotting.plot_signals(X, sample_freq_1, title_x)
# create a scatter plot of raw audio signals:
plotting.scatter_plot_signals(X, title_X, 'x')


# --------------------ICA ALGORITHM--------------------

# number of iterations to run FastICA:
num_iters = 100
print(len(X))
# center data:
X_center = ICA.center(X)
print(len(X_center))
print("\nSize of X_center: ", X_center.shape)
# whiten data:
X_white, whiten_filter = ICA.whiten(X_center)
print("Mean of whitened data:")
print(np.mean(X_white, axis=1))
print("Variance of whitened data:")
print(np.var(X_white, axis=1))

print(len(X_white))
print("\nSize of X_white: ", X_white.shape)

print(whiten_filter)
# run FastICA algorithm:
V = ICA.fastICA(X_white, num_sources=num_sig, num_iters=num_iters)
print("\nFinal rotation matrix V: ")
print(V)
print("\nSize of V: ", V.shape)
# recover source signals:
S = ICA.recover_sources(X_white, V, X, whiten_filter)
print("\nSize of S: ", S.shape)
# plot estimated source signals:
plotting.plot_signals(S, sample_freq_1, title_s)
# create a scatter plot of estimated source signals:
plotting.scatter_plot_signals(S, title_S, 's')



print("")


# --------------------PLAYING RESULTS--------------------

# convert source numpy arrays to .WAV files:
write(file_s1, sample_freq_1, S[0].astype(np.int16))
write(file_s2, sample_freq_2, S[1].astype(np.int16))

# """
# play audio files:
print("Playing", file_s1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_s2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s2)
play_obj = wave_obj.play()
play_obj.wait_done()
# """


# display plots:
plotter.show()

print("\n\nDone!\n")
