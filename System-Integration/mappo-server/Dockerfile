#Create a ubuntu base image with python 3 installed.
FROM python:3.8


# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp
COPY . /opt/webapp/

#ADD ./webapp/requirements.txt /tmp/requirements.txt

#RUN apt install python3-dev libpq-dev

RUN pip3 install -r requirements.txt

#$PORT is set by Heroku
CMD gunicorn --bind 0.0.0.0:$PORT wsgi