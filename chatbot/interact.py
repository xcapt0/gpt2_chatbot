import os
from itertools import chain

import torch
import torch.nn.functional as F

from utils import top_k_filter, lemma_sentence


class Chatbot:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def run(self):
        assert self.args['checkpoint'], 'Checkpoint was not found. Please specify the valid checkpoint through --checkpoint CHECKPOINT_PATH'
        self._load_checkpoint()

        print('Launching the chatbot...')
        print(f'If you want to stop, type the \"{self.args["stop_command"]}\" command')

        self.model.eval()

        with torch.no_grad():
            input_hists = []

            while True:
                sentence = input('You: ')
                if sentence == self.args['stop_command']:
                    print('Bot: Good bye.')
                    break

                sentence = lemma_sentence(sentence)

                input_ids = [self.args['sp1_id']] + self.tokenizer.encode(sentence)
                input_hists.append(input_ids)

                if len(input_hists) >= self.args['max_history']:
                    num_exceeded = len(input_hists) - self.args['max_history']
                    input_hists = input_hists[num_exceeded:]

                input_ids = [self.args['bos_id']] + list(chain.from_iterable(input_hists)) + [self.args['sp2_id']]
                start_sp_id = input_hists[0][0]
                next_sp_id = self.args['sp1_id'] if start_sp_id == self.args['sp2_id'] else self.args['sp2_id']
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(input_hists)) + [self.args['sp2_id']]
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)

                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args['device'])
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args['device'])

                output_ids = self._top_filtering(input_ids, token_type_ids)
                answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                print(f'Bot: {answer}')
                input_hists.append([self.args['sp2_id']] + self.tokenizer.encode(answer))

    def _top_filtering(self, input_ids, token_type_ids):
        output_ids = []

        for pos in range(self.args['max_len']):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0]

            logits = output[0, -1, :] / self.args['temperature']
            logits = top_k_filter(logits, top_k=self.args['top_k'])
            output = F.softmax(logits, dim=-1).unsqueeze(0)

            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_remove = cumsum_probs > self.args['top_p']
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)

            probs = torch.zeros(output.shape, device=self.args['device']).scatter_(-1, sorted_idxs, sorted_probs)
            idx = torch.multinomial(probs, 1)

            idx_item = idx.squeeze(-1).squeeze(-1).item()

            if idx_item in output_ids:
                continue

            output_ids.append(idx_item)

            if idx_item == self.args['eos_id']:
                break

            input_ids = torch.cat((input_ids, idx.reshape(1, 1)), dim=-1)
            next_type_id = torch.LongTensor([[self.args['sp2_id']]]).to(self.args['device'])
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape

        return output_ids

    def _load_checkpoint(self):
        path = self.args['checkpoint']
        if os.path.exists(path):
            print('Loading checkpoint...')
            checkpoint = torch.load(path, map_location=self.args['device'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Found checkpoint file: {os.path.basename(path)}')
        else:
            print("Can't find the specified checkpoint")
