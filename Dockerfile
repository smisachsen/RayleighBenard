FROM continuumio/miniconda3


WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

RUN apt install -y git tmux htop nano

RUN git clone https://github.com/smisachsen/RayleighBenard.git
WORKDIR RayleighBenard

RUN  git clone https://github.com/tensorforce/tensorforce.git \
    && cd tensorforce \
    && git checkout 0.5.0 -b tensorforce_0_5_0 

COPY script_launch_parallel.sh .
USER root