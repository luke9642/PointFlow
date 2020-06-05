FROM continuumio/anaconda3

COPY install.sh .

RUN apt-get install libgl1-mesa-glx libgomp1 -y && ./install.sh
