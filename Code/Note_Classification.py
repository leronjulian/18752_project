import librosa as lb 
import matplotlib.pyplot as plt 
import librosa.display as libDisp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np 

# Select Data
audio_name = 'Ashokan_Farewell'
# audio_name = 'Ghost_Riders'
# audio_name = 'Greensleeves'
# audio_name = 'The_Titanic'
# audio_name = 'Silent_Night'
X, sr = lb.load('../Data/Audio/' + audio_name + '.mp3')
Y = pd.read_csv('../Data/Audio/' + audio_name + '_gt.csv')


tempo, beat_frames = lb.beat.beat_track(y=X, sr=sr) # Gets the estimated tempo

MFCC = lb.feature.mfcc(X, sr=sr, hop_length=1024, htk=True) #(20, Num of segments)


# Visualize the MFCC
fig, ax = plt.subplots()
img = libDisp.specshow(MFCC, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.show()


# Locate note onset events by picking peaks in an onset strength envelope (Matches the # of GT notes)
o_env = lb.onset.onset_strength(X, sr=sr)
onset_frames = lb.onset.onset_detect(onset_envelope=o_env, sr=sr)
times = lb.times_like(o_env, sr=sr)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')
plt.xlabel('Time')
plt.ylabel('Onset Strength')
plt.legend() 
plt.show()

print(len(times[onset_frames]))
print(Y.shape)


# Convert Amplitude to spectrogram to dB-scaled spectrogram
D = np.abs(lb.stft(X))
libDisp.specshow(lb.amplitude_to_db(D, ref=np.max),
                         x_axis='time', y_axis='log')
plt.title('Power Spectrum')                        
plt.show()


# LDA
# clf = LinearDiscriminantAnalysis()
# clf.fit(MFCC, y)