import torch
import torch.nn
import librosa as rs
import soundfile as sf
from model import Net


# Load Model
m = Net()
m.load_state_dict(torch.load('dummy.pt'))


## Load input and transform to STFT
window = torch.hann_window(512)
x = rs.load("input.wav",sr=16000)[0]
x = torch.from_numpy(x)
x = x.unsqueeze(0)
X = torch.stft(x, n_fft=512, hop_length=128, win_length=512, window=window,return_complex=False)
X = torch.permute(X,(0,3,1,2))

# Run model
Y,h = m(X)


# Inverse STFT
Y = torch.permute(Y,(0,2,3,1))
y = torch.istft(Y[...,0] + Y[...,1]*1j, n_fft=512, hop_length=128, win_length=512, window=window)
y = y.detach().cpu().numpy()[0]

# Save
sf.write("output.wav",y,16000)  