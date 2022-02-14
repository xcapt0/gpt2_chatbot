FROM python:3.8.1
RUN mkdir /app
COPY . /app
WORKDIR /app/chatbot
RUN pip install -r requirements.txt
ENTRYPOINT ["bash"]