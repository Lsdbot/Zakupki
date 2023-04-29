FROM python:3.10-slim-buster
LABEL author="levsha-2001@mail.ru"


WORKDIR /app/backend
COPY . .


RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
