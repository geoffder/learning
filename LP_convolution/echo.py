import matplotlib.pyplot as plt
import numpy as np
import wave

from scipy.io.wavfile import write

spf = wave.open('helloworld.wav', 'r')
# get the audio into a numpy array
# sampling rate of 16kHz, bit rate = 16. Quantized with 2^16 possible values.
# encoded as a 16bit integer (amplitude)
signal = np.frombuffer(spf.readframes(-1), np.int16)
print('numpy signal shape:', signal.shape)

plt.plot(signal)
plt.title('hello world without echo')
plt.show()

delta = np.array([1, 0, 0])
noecho = np.convolve(signal, delta)
print('noecho signal shape:', noecho.shape)
# assert that the values are the same
# note that the lengths are not the same, so must slice noecho (it's longer)
assert(np.abs(noecho[:len(signal)] - signal).sum() < .0000001)

# MUST recast as 16bit integer. Otherwise will just hear a very loud noise
noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)  # specify sampling rate

filt = np.zeros(16000)  # 16k samples long, thus one second
filt[0] = 1  # repeat the signal itself
filt[4000] = .6  # decrease amplitude
filt[8000] = .3
filt[12000] = .2
filt[15999] = .1

plt.plot(filt)
plt.title('echo filter')
plt.show()

echo = np.convolve(signal, filt)
print('echo signal shape:', echo.shape)

echo = echo.astype(np.int16)
write('echo.wav', 16000, echo)

plt.plot(echo)
plt.title('hellow world with echo')
plt.show()
