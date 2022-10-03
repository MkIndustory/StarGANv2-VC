from sre_constants import NOT_LITERAL
from models import MappingNetwork
import torch
import coremltools
import numpy as np

mapping_network = MappingNetwork(16, 64, 20, hidden_dim=512).eval()

latent_dim = mapping_network.shared[0].in_features
label = torch.LongTensor([3])
example_inputs = (torch.randn(1, latent_dim), label)
traced = torch.jit.trace(mapping_network.forward, example_inputs)
traced.save("mapping_network.pt")

#print(torch.randn(1, latent_dim).shape)
#print(torch.randn(1, latent_dim).dtype)
#print(label.shape)

mlmodel = coremltools.converters.convert(
  traced,
  inputs=[
    coremltools.TensorType(shape=(1, 16), dtype=np.float32),#dtypeどうなる??
    coremltools.TensorType(shape=(1, ), dtype=np.int64)],#dtypeどうなる?
)

mlmodel.save("mapping_network.mlmodel")