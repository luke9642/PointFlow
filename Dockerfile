FROM continuumio/anaconda3

RUN conda create -n pytorch scikit-learn pandas numpy scipy tensorboard imageio

