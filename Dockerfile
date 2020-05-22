FROM continuumio/anaconda3

COPY install.sh .

RUN ./install.sh
