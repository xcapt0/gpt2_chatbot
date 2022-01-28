FROM python:3.8.1
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
WORKDIR /app/chatbot
ENTRYPOINT ["bash"]