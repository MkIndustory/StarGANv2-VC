from sre_constants import NOT_LITERAL
from models import Generator
import torch
import torchaudio
import librosa
import numpy as np
from Utils.JDC.model import JDCNet

generator = Generator(64, 64, 512, w_hpf=0, F0_channel=256)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

audio, source_sr = librosa.load("natsuVoice.wav", sr=24000)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32
source = preprocess(audio)

ref = torch.zeros([1, 64], dtype=torch.float32)




from typing import Optional , Union


# load F0 model

F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("bst.t7",map_location=torch.device('cpu'))['net']
F0_model.load_state_dict(params)
_ = F0_model.eval()

f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))

piki = Optional[torch.Tensor]
piki = None
#hh = Optional[torch.Tensor]()
#b: Union[torch.Tensor, None] #= None


#scripted_model = torch.jit.script(generator.forward)

example_inputs = [source.unsqueeze(1), ref, _ , f0_feat] # NONEがうまく変換できない。
traced = torch.jit.trace(generator, example_inputs)
#traced.save("generator.pt")

        