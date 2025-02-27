
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN set -xe && \
    apt-get -yqq update && \
    apt-get -yqq install python3-pip && \
    pip3 install --upgrade pip && \
    apt-get -yqq install python3-venv && \
    apt-get -yqq install rsync \
    apt-get -y install git

COPY requirements.txt.darwin /tmp/requirements.txt.darwin
#ENV PATH="/home/software/bin:${PATH}"
ENV STAGING_DIR=/staging/jaenmarquez

RUN pip3 install -r /tmp/requirements.txt.darwin