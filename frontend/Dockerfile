FROM python:3.10-slim-buster
LABEL author="levsha-2001@mail.ru"

WORKDIR /app/frontend
COPY . .


RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


EXPOSE 8501

ENTRYPOINT [ "streamlit", "run"]
CMD ["main.py"]
