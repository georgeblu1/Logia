FROM gcr.io/google-appengine/python:37
RUN virtualenv /env -p python3.7
ENV PATH /env/bin:$PATH
ADD requirements.txt requirements.txt
RUN /env/bin/pip install -r requirements.txt
ADD . /app
CMD gunicorn --timeout 180 -b 300
EXPOSE 8080