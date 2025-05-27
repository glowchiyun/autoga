FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales tzdata sudo
RUN apt-get -y install curl wget unzip git
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.9 python3.9-venv python3.9-dev python3-pip
RUN apt-get -y install libhdf5-serial-dev
#RUN update-alternatives --install /usr/bin/python3 python3 $(which python3.9) 1

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# aliases for the python system
ENV SPIP python3.9 -m pip
ENV SPY python3.9

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Setup HDF5 for installing `tables`
ENV HDF5_DIR /usr/lib/aarch64-linux-gnu/hdf5/serial

WORKDIR /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPIP install -U pip wheel
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/python3.9 -m pip
ENV PY /bench/venv/bin/python3.9 -W ignore
#RUN $PIP install -U pip==None wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN (grep -v '^\s*#' | xargs -L 1 $PIP install --no-cache-dir) < requirements.txt

RUN $PY runbenchmark.py autoga -s only

RUN /setup.sh


# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY runbenchmark.py $0 $*"]
CMD ["autoga", "test"]

