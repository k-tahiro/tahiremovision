FROM balenalib/raspberry-pi2-python:3.7.4


RUN [ "cross-build-start" ]
RUN apt update \
    && apt install -y \
    build-essential \
    libopenblas-dev \
    libblas-dev \
    m4 \
    cmake \
    cython \
    python3-dev \
    python3-yaml \
    python3-setuptools \
    python3-wheel \
    python3-pillow \
    python3-numpy
COPY [ "requirements.txt", "./" ]
RUN pip3 install --upgrade setuptools \
    && pip3 install -r requirements.txt \
    && pip3 install https://github.com/nmilosev/pytorch-arm-builds/raw/master/torch-1.2.0a0%2B8554416-cp37-cp37m-linux_armv7l.whl \
    && pip3 install https://github.com/nmilosev/pytorch-arm-builds/raw/master/torchvision-0.4.0a0%2Bd31eafa-cp37-cp37m-linux_armv7l.whl
RUN [ "cross-build-end" ]

COPY [ "config.py", "main.py", "predictor.py", "./" ]

ENTRYPOINT [ "uvicorn", "main:app" ]
CMD [ "--host", "0.0.0.0" ]

EXPOSE 8000
