FROM python:3.8
WORKDIR /app
COPY requirements.txt requirements.txt
COPY en_use_lg-0.4.3.tar.gz en_use_lg-0.4.3.tar.gz
RUN pip install -r requirements.txt
RUN pip install en_use_lg-0.4.3.tar.gz
RUN python -m spacy download en_core_web_lg
COPY . .
CMD [ "python", "app.py"]