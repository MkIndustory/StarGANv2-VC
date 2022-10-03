#pytorchレシピをとってくる
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
RUN apt-get update; exit 0
RUN apt-get -y install wget libsndfile1
RUN pip3 install SoundFile torchaudio munch parallel_wavegan torch pydub pyyaml click librosa coremltools
COPY . StarGANv2-VC
WORKDIR StarGANv2-VC
#RUN python3 tracing_style_encoder.py
#RUN python3 tracing_mapping_network.py
RUN python3 tracing_generator.py