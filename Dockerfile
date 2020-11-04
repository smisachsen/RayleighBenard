FROM continuumio/miniconda3


WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

RUN apt install -y git tmux htop nano

RUN git clone https://github.com/smisachsen/RayleighBenard.git
WORKDIR RayleighBenard

COPY run_single.py .
