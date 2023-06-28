# use nvidia container to enable gpu/cuda usage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

USER root

COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# install missing dependencies
RUN apt-get update && apt-get install -y python3-pip git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install all pip packages
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  
RUN pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu121 
RUN pip3 install setuptools==59.5.0
RUN pip3 install opencv-python
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install onnxruntime
RUN pip3 install onnx
RUN pip3 install ipykernel
RUN pip3 install transformers
RUN pip3 install addict
RUN pip3 install yapf
RUN pip3 install timm
RUN pip3 install loguru
RUN pip3 install tqdm
RUN pip3 install scikit-image
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install tensorboard
RUN pip3 install seaborn
RUN pip3 install open_clip_torch
RUN pip3 install SciencePlots
RUN pip3 install einops
RUN pip3 install gradio
RUN pip3 install opencv-python
RUN pip3 install -U diffusers[torch]
RUN pip3 install gputil

# also install vim because it is the best editor ofc
RUN apt-get install -y vim
