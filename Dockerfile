FROM continuumio/miniconda3

RUN apt install -y git tmux


WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml


RUN git clone https://github.com/smisachsen/RayleighBenard.git
WORKDIR RayleighBenard

COPY main.py .
