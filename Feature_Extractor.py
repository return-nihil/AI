'''
FEATURE EXTRACTORS.

Functions for: 
- mel spectrogram, 
- Δ, 
- ΔΔ, 
- chromagram, 
- contrast, 
- tonnetz, 
- mfcc.
'''



import librosa
import torch
import torchvision.transforms.functional as functional
import torchaudio
import numpy as np




def get_mel(device, 
            sig, 
            sr = 16000, 
            n_fft = 2048,
            n_mels = 64, 
            normalize = True, 
            size = ([256, 256])):

  spec = librosa.feature.melspectrogram(y = sig, sr = sr, n_fft = n_fft, n_mels = n_mels, fmax = sr/2)
  spec = np.log(spec + 1e-9)

  if normalize == True:
    spec = librosa.util.normalize(spec)

  spec = torch.unsqueeze(torch.from_numpy(spec), 0).to(device)
  spec = functional.resize(spec, size)

  return spec


def get_delta(device, 
              sig, 
              sr = 16000, 
              n_fft = 2048,
              n_mels = 64, 
              normalize = True, 
              size = ([256, 256])):

  spec = librosa.feature.melspectrogram(y = sig, sr = sr, n_fft = n_fft, n_mels = n_mels, fmax = sr/2)
  spec = np.log(spec + 1e-9)
  delta = librosa.feature.delta(data = spec)

  if normalize == True:
    spec = librosa.util.normalize(delta)

  delta = torch.unsqueeze(torch.from_numpy(delta), 0).to(device)
  delta = functional.resize(delta, size)

  return delta


def get_delta_delta(device, 
                    sig, 
                    sr = 16000, 
                    n_fft = 2048,
                    n_mels = 64, 
                    normalize = True, 
                    size = ([256, 256])):

  spec = librosa.feature.melspectrogram(y = sig, sr = sr, n_fft = n_fft, n_mels = n_mels, fmax = sr/2)
  spec = np.log(spec + 1e-9)
  delta_delta = librosa.feature.delta(data = spec, order = 2)

  if normalize == True:
    spec = librosa.util.normalize(delta_delta)

  delta_delta = torch.unsqueeze(torch.from_numpy(delta_delta), 0).to(device)
  delta_delta = functional.resize(delta_delta, size)

  return delta_delta


def get_chroma(device, 
               sig, 
               sr = 16000, 
               n_fft = 2048, 
               hop_length = 512, 
               n_chroma = 128, 
               normalize = True, 
               size = ([256, 256])):

  chroma = librosa.feature.chroma_stft(sig = sig, sr = sr, n_fft = n_fft, hop_length = hop_length, n_chroma = n_chroma)

  if normalize == True:
    chroma = librosa.util.normalize(chroma)

  chroma = torch.unsqueeze(torch.from_numpy(chroma), 0).to(device)
  chroma = functional.resize(chroma, size)

  return chroma


def get_contr(device, 
              sig,
              sr = 16000, 
              normalize = True, 
              size = ([256, 256])):

  contr = librosa.feature.contrast(y = sig, sr = sr)

  if normalize == True:
    contr = librosa.util.normalize(contr)

  contr = torch.unsqueeze(torch.from_numpy(contr), 0).to(device)
  contr = functional.resize(contr, size)

  return contr


def get_tonnetz(device, 
               sig,
               sr = 16000, 
               normalize = True, 
               size = ([256, 256])):

  tonnetz = librosa.feature.tonnetz(y = sig, sr = sr)

  if normalize == True:
    tonnetz = librosa.util.normalize(tonnetz)

  tonnetz = torch.unsqueeze(torch.from_numpy(tonnetz), 0).to(device)
  tonnetz = functional.resize(tonnetz, size)

  return tonnetz


def get_mfcc(device, 
             sig,
             sr = 16000, 
             n_mfcc = 16,
             normalize = True, 
             size = ([256, 256])):

  mfcc = librosa.feature.mfcc(y = sig, sr = sr, n_mfcc = n_mfcc)

  if normalize == True:
    mfcc = librosa.util.normalize(mfcc)

  mfcc = torch.unsqueeze(torch.from_numpy(mfcc), 0).to(device)
  mfcc = functional.resize(mfcc, size)

  return mfcc
