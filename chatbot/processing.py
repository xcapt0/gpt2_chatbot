from datasets import load_dataset
from tqdm.auto import tqdm


class Processing:
    def __init__(self, tokenizer, train_frac):
        self.tokenizer = tokenizer
        self.train_frac = train_frac

    def _load_daily(self):
        dataset = load_dataset('daily_dialog')
        train_dialogues = dataset['train']['dialog']
        valid_dialogues = dataset['validation']['dialog']
        test_dialogues = dataset['test']['dialog']

        all_dialogues = train_dialogues + valid_dialogues + test_dialogues

        for i, dialogue in enumerate(tqdm(all_dialogues)):
            new_dialogue = []
            for sentence in dialogue:
                token_list = self.tokenizer.tokenize(sentence.strip().replace('’', '\''))
                token_list = self._process_token_list(token_list)
                text = self.tokenizer.convert_tokens_to_string(token_list)
                new_dialogue.append(text)

            all_dialogues[i] = new_dialogue

        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues

    def _load_empathetic(self):
        dataset = load_dataset('empathetic_dialogues')
        train_data = dataset['train']
        valid_data = dataset['validation']
        test_data = dataset['test']

        sentences = train_data['utterance'] + valid_data['utterance'] + test_data['utterance']
        total_conv_ids = train_data['conv_id'] + valid_data['conv_id'] + test_data['conv_id']
        total_speaker_ids = train_data['speaker_idx'] + valid_data['speaker_idx'] + test_data['speaker_idx']

        conv_dict = {}
        cur_speaker_idx = -1
        for i, sentence in enumerate(tqdm(sentences)):
            conv_id = total_conv_ids[i]
            speaker_idx = total_speaker_ids[i]

            sentence_modified = sentence.strip().replace('_comma_', ',')
            new_token_list = self._process_token_list(self.tokenizer.tokenize(sentence_modified))
            text = self.tokenizer.convert_tokens_to_string(new_token_list)

            if '_conv' in sentence:
                continue

            if conv_id not in conv_dict:
                conv_dict[conv_id] = []
                cur_speaker_idx = -1

            if cur_speaker_idx != speaker_idx:
                conv_dict[conv_id].append(text)
                cur_speaker_idx = speaker_idx
            else:
                conv_dict[conv_id][-1] += f" {text}"

        train_dialogues = []
        valid_dialogues = []

        train_dialogue_num = int(len(conv_dict) * self.train_frac)
        for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
            if i < train_dialogue_num:
                train_dialogues.append(utter_list)
            else:
                valid_dialogues.append(utter_list)

        return train_dialogues, valid_dialogues

    def _load_persona(self):
        import requests

        url = 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json'
        response = requests.get(url)
        assert response.status_code == 200, 'Error receiving data from server'
        dataset = response.json()

        train_data = dataset['train']
        valid_data = dataset['valid']
        all_data = train_data + valid_data
        all_dialogues = []

        for obj in tqdm(all_data):
            dialogue = obj['utterances'][-1]['history']
            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                if sentence.strip() != '__ SILENCE __':
                    token_list = self.tokenizer.tokenize(sentence.strip())
                    new_token_list = self._process_token_list(token_list)
                    text = self.tokenizer.convert_tokens_to_string(new_token_list)
                    new_dialogue.append(text)

            all_dialogues.append(new_dialogue)

        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues

    def _load_blended(self):
        dataset = load_dataset('blended_skill_talk')
        data_train = dataset['train']
        data_valid = dataset['validation']
        data_test = dataset['test']

        all_previous_sentences = data_train['previous_utterance'] + \
                                 data_valid['previous_utterance'] + \
                                 data_test['previous_utterance']
        all_free_messages = data_train['free_messages'] + \
                            data_valid['free_messages'] + \
                            data_test['free_messages']
        all_guided_messages = data_train['guided_messages'] + \
                              data_valid['guided_messages'] + \
                              data_test['guided_messages']

        all_dialogues = []
        for i, free_message in enumerate(tqdm(all_free_messages)):
            free_message_list = [sentence.strip() for sentence in free_message if len(sentence.strip()) > 0]
            guided_message_list = [sentence.strip() for sentence in all_guided_messages[i] if len(sentence.strip()) > 0]
            dialogue = all_previous_sentences[i]

            for j in range(len(free_message_list)):
                token_list = self._process_token_list(self.tokenizer.tokenize(free_message_list[j]))
                text = self.tokenizer.convert_tokens_to_string(token_list)
                dialogue.append(text)

                if j < len(guided_message_list):
                    token_list = self._process_token_list(self.tokenizer.tokenize(guided_message_list[j]))
                    text = self.tokenizer.convert_tokens_to_string(token_list)
                    dialogue.append(text)

            all_dialogues.append(dialogue)

        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues

    @staticmethod
    def _process_token_list(token_list):
        space = 'Ġ'
        quotes = ['"', '\'']
        end_marks = ['.', ',', '?', '!', '...']
        abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
        token_list[0] = token_list[0].capitalize()
        quote_count = 0

        for i, token in enumerate(token_list):
            if space in token:
                if token[1:] in end_marks or token[1:] in abbreviations:
                    token_list[i] = token[1:]

                if token[1:] == quotes[1]:
                    if i < len(token_list) - 1:
                        if token_list[i + 1] in abbreviations or (
                                token_list[i + 1][0] == space and token_list[i + 1][1:] in abbreviations):
                            token_list[i] = token[1:]

            if token[0] == space and token[1:] in quotes:
                if quote_count % 2 == 1:
                    token_list[i] = token[1:]
                    quote_count = 0
                else:
                    if i < len(token_list) - 1 and token_list[i + 1][0] == space:
                        token_list[i + 1] = token_list[i + 1][1:]
                    quote_count += 1

            if token in end_marks or token[1:] in end_marks:
                if i < len(token_list) - 1:
                    if token_list[i + 1][0] != space:
                        token_list[i + 1] = space + token_list[i + 1].capitalize()
                    else:
                        token_list[i + 1] = space + token_list[i + 1][1:].capitalize()

        new_token_list = [token for token in token_list if token != space and len(token) > 0]
        if new_token_list[-1] not in end_marks:
            new_token_list.append(end_marks[0])

        return new_token_list
