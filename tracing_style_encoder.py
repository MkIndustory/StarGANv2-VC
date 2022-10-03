from sre_constants import NOT_LITERAL
from models import StyleEncoder
import torch
import torchaudio
import coremltools
import librosa
import numpy as np

style_encoder = StyleEncoder(64, 64, 20, 512).eval()
#style_encoder = StyleEncoder(64, 64, 16, 20).eval()

mean, std = -4, 4
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

wave, sr = librosa.load("natsuVoice.wav", sr=24000)
audio, index = librosa.effects.trim(wave, top_db=30)

wave_tensor =  wave_tensor = torch.from_numpy(wave).float()
#print(wave_tensor.shape)
mel_tensor = to_mel(wave_tensor)
mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
#print(mel_tensor.shape)

label = torch.LongTensor([3])
example_inputs = (mel_tensor.unsqueeze(1), label)
traced = torch.jit.trace(style_encoder.forward, example_inputs)
traced.save("style_encoder.pt")

#print(mel_tensor.unsqueeze(1).shape)

#test = style_encoder.forward(mel_tensor.unsqueeze(1), label)
#print(test.shape) #torch.Size([1, 64])
#print(test.dtype) #torch.float32

mlmodel = coremltools.converters.convert(
  traced,
  inputs=[
    coremltools.TensorType(shape=(1, 1, 80, 7841), dtype=np.float32),
    coremltools.TensorType(shape=(1, ), dtype=np.int64)],
)
#print(traced)
mlmodel.save("style_encoder.mlmodel")