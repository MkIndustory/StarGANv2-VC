from sre_constants import NOT_LITERAL
from models import Generator
import torch
import torchaudio
import librosa
import coremltools
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
example_inputs = [source.unsqueeze(1), ref, f0_feat]
traced = torch.jit.trace(generator, example_inputs)
traced.save("generator.pt")

#print(source.unsqueeze(1).shape, ref.shape, f0_feat.shape)
#print(source.unsqueeze(1).dtype, ref.dtype, f0_feat.dtype)

mlmodel = coremltools.converters.convert(
  traced,
  inputs=[
    coremltools.TensorType(shape=(1, 1, 80, 7841), dtype=np.float32),
    coremltools.TensorType(shape=(1, 64), dtype=np.float32),
    coremltools.TensorType(shape=(1, 256, 10, 7841), dtype=np.float32)]
)
mlmodel.save("generator.mlmodel")

        