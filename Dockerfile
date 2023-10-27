FROM python:3.11

RUN apt-get update && apt-get install -y jq

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython

RUN mkdir /app
COPY src/ /app/src/

COPY pyproject.toml setup.py LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app
