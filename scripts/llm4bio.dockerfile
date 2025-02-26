
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN set -xe && apt-get -yqq update && apt-get -yqq install python3-pip && pip3 install --upgrade pip && apt-get -yqq install rsync

COPY requirements.txt /tmp/requirements.txt

#ENV PATH="/home/software/bin:${PATH}"

RUN pip3 install -r /tmp/requirements.txt