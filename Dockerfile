ARG TENSORFLOW="2.3.1-gpu"

FROM tensorflow/tensorflow:${TENSORFLOW}

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/dacon-ai/NIA-Landmark.git

WORKDIR /NIA-Landmark/Recognition

RUN pip install -r requirements.txt
