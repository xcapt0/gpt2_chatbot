# 🍵 GPT2 Chatbot

GPT-2 chatbot for daily conversations trained on `Daily Dialogue`, `Empathetic Dialogues`, `PERSONA-CHAT`, `Blended Skill Talk` datasets. This chatbot is made based on GPT2 Model transformer with a language modeling head on top.

![chatbot](https://user-images.githubusercontent.com/70326958/151570518-ce70261a-6e8e-47a0-92e5-2d7638e7aa68.jpg)


## ⌛ Installation

Download the [model](https://gpt2chatbot.s3.us-east-2.amazonaws.com/model.h5) from AWS S3 storage and run the following command:

```sh
git pull https://github.com/xcapt0/gpt2_chatbot.git
docker build -t gpt2_bot .
```

## 🤖 Usage

Run the docker container:
```sh
docker run --rm -it gpt2_bot
```

There are 2 different ways to use the chatbot: `train` and `interact` mode

### Interaction mode
To launch the chatbot run the following command. Specify `--checkpoint` path to your model
```sh
python chatbot.py --mode interact --checkpoint path/to/model.h5
```

### Train mode
To train the model run the following command. Specify `--checkpoint` if you needed
```sh
python chatbot.py --mode train
```

## 📝 License

Copyright © 2022 [Vadim Mukhametgareev](https://github.com/xcapt0).<br />
This project is [MIT](https://github.com/xcapt0/gpt2_chatbot/blob/master/LICENSE) licensed.
