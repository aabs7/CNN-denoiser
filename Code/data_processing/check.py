from data_processing.urban_sound import UrbanSound8K
from data_processing.feature_extractor import FeatureExtractor
from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from utils import read_audio
import numpy as np
import librosa

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

urbansound_basepath = '/home/abhish/Documents/Dataset/UrbanSound8K'
mozilla_basepath = '/home/abhish/Documents/Dataset/MozillaDataset'

mcv = MozillaCommonVoiceDataset(mozilla_basepath,val_dataset_size = 1000)
clean_train_filenames,clean_val_filenames = mcv.get_train_val_filenames()

clean_filename = np.random.choice(clean_train_filenames)

clean_audio,sr = read_audio(clean_filename,config['fs'])
print("clean audio:",clean_audio)

#us8k = UrbanSound8K(urbansound_basepath,val_dataset_size = 200)
#noise_train_filenames,noise_val_filenames = us8k.get_train_val_filenames()

#print(noise_train_filenames)

# 
# noise_filename = np.random.choice(noise_train_filenames)
# 
# noise_audio, sr = read_audio(noise_filename,config['fs'])
# print(noise_audio)

# print("Sample rate:",sr)
# print("Size of audio with silent frame:",np.size(noise_audio))
# 
# def remove_silent_frame(audio):
    # trimed_audio = []
    # indices = librosa.effects.split(audio,hop_length = config["overlap"],top_db = 20)
# 
    # for index in indices:
        # trimed_audio.extend(audio[index[0]:index[1]])
# 
    # return np.array(trimed_audio)
# 
# remove silent frame from noise audio
# noise_audio = remove_silent_frame(noise_audio)
# print("Size of audio without silent frame:",np.size(noise_audio))
# 
# noise_audio_fe = FeatureExtractor(noise_audio,windowLength = windowLength,overlap = config["overlap"],
                    # sample_rate = sr)
# 
# spectogram = noise_audio_fe.get_stft_spectrogram()
# 
# print(spectogram)


