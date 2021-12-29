import os
import math
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data import DialoguesDataset
from utils import PadCollate


class Trainer:
    def __init__(self, model, args):
        print('Loading the optimizer...')
        self.optimizer = AdamW(model.parameters(), lr=args['lr'])
        self.best_loss = 1e+10
        self.last_epoch = 0

        print('Loading train & valid data...')
        train_dataset = DialoguesDataset('train', args)
        valid_dataset = DialoguesDataset('valid', args)
        pad = PadCollate(args)

        self.train_loader = DataLoader(train_dataset,
                                       collate_fn=pad,
                                       shuffle=True,
                                       batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       collate_fn=pad,
                                       batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)

        if not os.path.exists(args['models_dir']):
            os.makedirs(args['models_dir'])

        # Calculate total training steps
        num_batches = len(self.train_loader)
        total_train_steps = args['num_epochs'] * num_batches
        warmup_steps = int(args['warmup_ratio'] * total_train_steps)

        self.model = model
        self.args = args
        self.scheduler = get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2
        )

        if args['checkpoint']:
            self._load_checkpoint()

    def train(self):
        print('Launch training...')

        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, start_epoch + self.args['num_epochs']):
            print('-' * 50 + f'\nEpoch: {epoch}\n' + '-' * 50)

            self.model.train()
            train_losses = []
            train_perplexity = []

            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.args['device'])
                token_type_ids = token_type_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss, logits = outputs[0], outputs[1]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                train_perplexity.append(ppx)

            train_losses = [loss.item() for loss in train_losses]
            train_perplexity = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in train_perplexity]
            train_loss = np.mean(train_losses)
            train_ppx = np.mean(train_perplexity)
            print(f'Train loss: {train_loss} \nTrain perplexity: {train_ppx}')

            self.last_epoch += 1

            valid_loss, valid_ppx = self.validate()

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'epoch': self.last_epoch
                }

                filename = f"{self.args['model_dir']}/model_best_{round(self.best_loss, 4)}.h5"
                torch.save(state_dict, filename)
                print(f'Checkpoint saved: {filename}')

            print(f'Best valid loss: {self.best_loss}')
            print(f'Valid loss: {valid_loss} \nValid perplexity: {valid_ppx}')

        print('Training completed')

    def validate(self):
        print('Launch validation...')
        self.model.eval()

        valid_losses = []
        valid_ppxs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.args['device'])
                token_type_ids = token_type_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss, logits = outputs[0], outputs[1]

                valid_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                valid_ppxs.append(ppx)

            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppxs = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in valid_ppxs]
            valid_loss = np.mean(valid_losses)
            valid_ppx = np.mean(valid_ppxs)

            if math.isnan(valid_ppx):
                valid_ppx = 1e+8

        return valid_loss, valid_ppx

    def _load_checkpoint(self):
        path = self.args['checkpoint']
        if os.path.exists(path):
            print('Loading checkpoint...')
            checkpoint = torch.load(path, map_location=self.args['device'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

            print(f'The training restarts with the specified checkpoint: {os.path.basename(path)}')
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
            self.last_epoch = checkpoint['epoch']
        else:
            print("Can't find the specified checkpoint")