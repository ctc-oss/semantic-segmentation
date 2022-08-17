# FROM python:3.8
FROM conda/miniconda3

RUN apt update -y && apt install -y curl unzip ffmpeg libsm6 libxext6

## AWS CLI setup
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
  && unzip awscliv2.zip \
  && ./aws/install

# Environment setup
# COPY requirements.txt ./
# RUN pip3 install -r requirements.txt
COPY environment.yml ./
# RUN sed -i 's/gdal_python36/base/g' environment.yml
RUN conda env update -f environment.yml
RUN conda init bash
RUN /bin/bash -c "source activate gdal_python36"
RUN echo "source activate gdal_python36" >> ~/.bashrc
RUN mkdir -p ~/.aws/ \
  && echo "[default]" > ~/.aws/config \
  && echo "region = us-east-2" >> ~/.aws/config
