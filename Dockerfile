ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG GITHUB_USER
ARG DEV_SOURCE
ARG GITHUB_TOKEN

WORKDIR /src

# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials

# clone projects from public/private github repos
RUN git clone --depth 1 --branch challenge https://github.com/${DEV_SOURCE}/data_port.git &&\
    git clone --depth 1 --branch inception_loops https://github.com/${DEV_SOURCE}/nnvision.git &&\
    git clone -b transformer_readout https://github.com/KonstantinWilleke/neuralpredictors

FROM ${BASE_IMAGE}
COPY --from=base /src /src

# lines below are necessasry to fix an issue explained here: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install screen
RUN apt-get -y update && apt-get install -y \
    screen

# install third-party libraries
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install \
    wandb \
    moviepy \
    imageio \
    # git+https://github.com/sinzlab/sensorium@main \
    git+https://github.com/mohammadbashiri/classicalv1@master \
    nnfabrik

# install the cloned repos
RUN python -m pip install -e /src/data_port &&\
    python -m pip install --no-use-pep517 -e /src/neuralpredictors &&\
    python -m pip install -e /src/nnvision &&\
    python -m pip install git+https://github.com/sacadena/ptrnets &&\
    python -m pip install git+https://github.com/dicarlolab/CORnet


# install the current project
WORKDIR /project
RUN mkdir /project/invariant
COPY ./invariant /project/invariant
COPY ./setup.py /project
RUN python -m pip install -e /project
