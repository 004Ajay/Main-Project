FROM python:3.10.11

WORKDIR /Main

COPY . . 

RUN pip install -r requirements.txt