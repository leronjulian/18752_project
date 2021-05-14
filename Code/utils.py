from itertools import tee
from scipy import signal
from scipy.io.wavfile import read as wavfile_read

import librosa
import matplotlib.pyplot as plt
import numpy as np

# use wavfile instead, much simpler
def load_wav(fname, to_mono=True, to_float=False):
    sample_rate, data = wavfile_read(fname)
    print('sample rate: ', sample_rate)

    if to_mono:
        data = data.mean(axis=1)
    if to_float:
        data = data.astype(float)
    # normalize to between 0 and 1
    data = data / (2**16)
    return sample_rate, data

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# spectrogram plot
def get_time_interval(data, sr, time_interval):
    assert len(time_interval) == 2, "time interval specifies the region of recording to get (should be len 2)"
    sample_interval = [b*sr for b in time_interval]
    start, end = sample_interval[0], sample_interval[1]
    return data[start: end]
    
def spectral_analysis(notes_data, sr, win_length=2048, time_interval=None):
    if time_interval is not None:
        notes_data = get_time_interval(notes_data, sr, time_interval)
        
    # pick a longer win_length than default (i.e. 2048) so that we get better frequency resolution
    X = librosa.stft(notes_data, n_fft=win_length, win_length=win_length)
    
    Xdb = librosa.amplitude_to_db(abs(X))
    return Xdb
    
def plot_spectrogram(spec_data, sr, win_length, take_log=False, ylim=None, ax=None):
    plt.figure(figsize=(14, 5))
    
    hop_length = win_length // 4
    if not take_log:
        librosa.display.specshow(spec_data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    else:
        librosa.display.specshow(spec_data, sr=sr, hop_length=hop-length, x_axis='time', y_axis='log', ax=ax)
    
    if ax is not None:
        if ylim is not None:
            ax.set_ylim(0, ylim)
#         ax.colobar()
    else:
        if ylim is not None:
            plt.ylim(0, ylim)
        plt.colorbar()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(cutoffs, fs, order=5):
    assert len(cutoffs) == 2
    assert cutoffs[0] < cutoffs[1]
    nyq = 0.5 * fs
    low_normal_cutoff = cutoffs[0] / nyq
    high_normal_cutoff = cutoffs[1] / nyq
    b, a = signal.butter(order, [low_normal_cutoff, high_normal_cutoff], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoffs, fs, order=5):
    assert cutoffs[0] < cutoffs[1]
    b, a = butter_bandpass(cutoffs, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def bandpass_filter(audio, sr, low_bound, high_bound, order=5, vis_freq_resp=True):
    # filter with high pass filter so that notes above 262Hz (middle-C) are preserved

    nyq_freq = sr / 2.0
    order = 5
    b, a = butter_bandpass([low_bound, high_bound], sr, order=order)

    if vis_freq_resp:
        w, h = signal.freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * sr * w / np.pi, np.abs(h), 'b')
        plt.plot(high_bound, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(high_bound, color='k')
        plt.xlim(0, 2000)
        plt.title("High Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

    filtered = signal.filtfilt(b, a, audio)
    filtered = filtered[~np.isnan(filtered)]

    return filtered
