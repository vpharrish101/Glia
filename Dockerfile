FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=secret,id=pip_ca \
    if [ -r /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu' -r requirements.txt \
    && pip check

COPY src ./src
COPY config ./config

CMD ["python","-m","src.main","inference"]
