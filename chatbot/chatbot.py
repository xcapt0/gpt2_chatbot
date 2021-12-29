import yaml
import torch
import nltk
from glob import glob
from argparse import ArgumentParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from data import Dialogues
from utils import set_seed


def main(args):
    set_seed(args['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['device'] = device

    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer, device)

    if dataset_is_missing(args):
        dialogues = Dialogues(tokenizer, args)
        train_dataset, valid_dataset = dialogues.load()

        dataset_types = ['train', 'valid']
        datasets = [train_dataset, valid_dataset]

        for dataset_type, dataset in zip(dataset_types, datasets):
            dialogues.save(dataset_type, tokenizer, dataset)

    if args['mode'] == 'train':
        from train import Trainer
        trainer = Trainer(model, args)
        trainer.train()
    elif args['mode'] == 'interact':
        from interact import Chatbot
        chatbot = Chatbot(model, tokenizer, args)
        chatbot.run()


def dataset_is_missing(args):
    if len(glob(f'{args["dataset_dir"]}/*.pickle')) == 0:
        return True
    return False


def load_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args['model_name'])
    special_tokens = ['<speaker1>', '<speaker2>']
    tokenizer.add_special_tokens({
        'bos_token': '<bos>',
        'additional_special_tokens': special_tokens
    })

    # add new token ids to args
    special_tokens += ['<bos>', '<eos>']
    sp1_id, sp2_id, bos_id, eos_id = tokenizer.encode(special_tokens)
    args['sp1_id'] = sp1_id
    args['sp2_id'] = sp2_id
    args['bos_id'] = bos_id
    args['eos_id'] = eos_id

    return tokenizer


def load_model(args, tokenizer, device):
    model = GPT2LMHeadModel.from_pretrained(args['model_name']).to(device)
    model.resize_token_embeddings(len(tokenizer))
    return model


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help='Pass "train" for training mode and "interact" for interaction mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint of the model')

    user_args = parser.parse_args()
    arguments = yaml.safe_load(open('config.yml'))
    arguments.update(vars(user_args))

    main(arguments)
