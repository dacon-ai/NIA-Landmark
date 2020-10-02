#! /bin/bash

sudo docker run -it --name nia-test -v /home/ubuntu/Dacon/cpt_data/landmark:/tmp --gpus all nia-landmark
 
