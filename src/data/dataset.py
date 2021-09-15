import torch

import src.commons.utils as utils

from torch.utils.data import Dataset
from transformers import BertTokenizer


class NERDatasetBase(Dataset):
    def __init__(self, text_dir, aux_text_dir, tokenizer: BertTokenizer, label_scheme):
        self.ids, self.tokens, self.labels, self.image_ids, self.aux_text = utils.read_text_as_list(text_dir, aux_text_dir)

        self.tokenizer = tokenizer
        self.index_map = dict(enumerate(label_scheme))
        self.label_map = {l: i for i, l in self.index_map.items()}

        self._prepare_encoding_fields_from_start()

    def _prepare_encoding_fields_from_start(self):
        self.tokenized = []
        self.input_ids = []
        self.input_msk = []
        self.segment_ids = []
        self.label_ids = []
        self.label_msk = []

        for i in range(len(self.ids)):
            tokens = self.tokens[i]
            labels = self.labels[i]
            aux_text = self.aux_text[i] if self.aux_text is not None else None

            tokenized, input_ids, input_msk, segment_ids, label_ids, label_msk = process_sample(self.tokenizer, tokens, labels, self.label_map, aux_text)

            self.tokenized.append(tokenized)
            self.input_ids.append(input_ids)
            self.input_msk.append(input_msk)
            self.segment_ids.append(segment_ids)
            self.label_ids.append(label_ids)
            self.label_msk.append(label_msk)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def collate(self, batch, pad_tok=0):
        raise NotImplementedError()


class NERDataset(NERDatasetBase):
    def __init__(self, text_dir, image_dir, aux_text_dir, tokenizer: BertTokenizer, label_scheme):

        super().__init__(text_dir, aux_text_dir, tokenizer, label_scheme)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        input_msk = self.input_msk[index]
        segment_ids = self.segment_ids[index]
        label_ids = self.label_ids[index]
        label_msk = self.label_msk[index]
        
        return input_ids, input_msk, segment_ids, label_ids, label_msk

    def collate(self, batch, pad_tok=0):
        # Unwrap the batch into every field
        input_ids, input_mask, segment_ids, label_ids, label_mask = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_mask, p_segment_ids, p_label_ids, p_label_mask = [], [], [], [], []

        # How much padding do we need?
        max_seq_length = max(map(len, input_ids))

        for i in range(len(input_ids)):
            padding_length = max_seq_length - len(input_ids[i])

            p_input_ids.append(input_ids[i] + [pad_tok] * padding_length)
            p_input_mask.append(input_mask[i] + [pad_tok] * padding_length)
            p_segment_ids.append(segment_ids[i] + [pad_tok] * padding_length)
            p_label_ids.append(label_ids[i] + [pad_tok] * padding_length)
            p_label_mask.append(label_mask[i] + [pad_tok] * padding_length)

        batch_dict = {
            'input_ids': torch.tensor(p_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(p_input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(p_segment_ids, dtype=torch.long),
            'visual_embeddings': None,
            'visual_position_ids': None,
            'visual_embeddings_type': None,
            'labels': torch.tensor(p_label_ids, dtype=torch.long),
            'label_mask': torch.tensor(p_label_mask, dtype=torch.long)
        } 

        return batch_dict


class NERDatasetWithGloablImageFeatures(NERDatasetBase):
    def __init__(self, text_dir, image_dir, aux_text_dir, tokenizer: BertTokenizer, label_scheme):

        super().__init__(text_dir, aux_text_dir, tokenizer, label_scheme)

        self.images, self.objects = utils.read_global_image_features_as_list(self.image_ids, image_dir)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        input_msk = self.input_msk[index]
        segment_ids = self.segment_ids[index]
        label_ids = self.label_ids[index]
        label_msk = self.label_msk[index]
        image_ids = self.images[index]
        
        return input_ids, input_msk, segment_ids, label_ids, label_msk, image_ids

    def collate(self, batch, pad_tok=0):
        # Unwrap the batch into every field
        input_ids, input_mask, segment_ids, label_ids, label_mask, image_ids = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_mask, p_segment_ids, p_label_ids, p_label_mask, p_image_ids = [], [], [], [], [], []

        # How much padding do we need?
        max_seq_length = max(map(len, input_ids))

        for i in range(len(input_ids)):
            padding_length = max_seq_length - len(input_ids[i])

            p_input_ids.append(input_ids[i] + [pad_tok] * padding_length)
            p_input_mask.append(input_mask[i] + [pad_tok] * padding_length)
            p_segment_ids.append(segment_ids[i] + [pad_tok] * padding_length)
            p_label_ids.append(label_ids[i] + [pad_tok] * padding_length)
            p_label_mask.append(label_mask[i] + [pad_tok] * padding_length)
        
        p_input_ids = torch.tensor(p_input_ids, dtype=torch.long)
        p_input_mask = torch.tensor(p_input_mask, dtype=torch.long)
        p_segment_ids = torch.tensor(p_segment_ids, dtype=torch.long)
        p_image_ids = torch.cat(image_ids, dim=0)
        p_label_ids = torch.tensor(p_label_ids, dtype=torch.long)
        p_label_mask = torch.tensor(p_label_mask, dtype=torch.long)

        batch_dict = {
            'input_ids': p_input_ids, 'attention_mask': p_input_mask, 'token_type_ids': p_segment_ids,
            'visual_embeddings': p_image_ids, 'visual_position_ids': None, 'visual_embeddings_type': None,
            'labels': p_label_ids, 'label_mask': p_label_mask
        } 

        return batch_dict


class NERDatasetWithRegionalImageFeatures(NERDatasetBase):
    def __init__(self, text_dir, image_dir, aux_text_dir, tokenizer: BertTokenizer, label_scheme):

        super().__init__(text_dir, aux_text_dir, tokenizer, label_scheme)

        self.images, self.objects = utils.read_regional_image_features_as_list(self.image_ids, image_dir)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        input_msk = self.input_msk[index]
        segment_ids = self.segment_ids[index]
        label_ids = self.label_ids[index]
        label_msk = self.label_msk[index]
        image_ids = self.images[index]
        
        return input_ids, input_msk, segment_ids, label_ids, label_msk, image_ids

    def collate(self, batch, pad_tok=0):
        # Unwrap the batch into every field
        input_ids, input_mask, segment_ids, label_ids, label_mask, image_ids = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_mask, p_segment_ids, p_label_ids, p_label_mask, p_image_ids = [], [], [], [], [], []

        # How much padding do we need?
        max_seq_length = max(map(len, input_ids))

        for i in range(len(input_ids)):
            padding_length = max_seq_length - len(input_ids[i])
            p_input_ids.append(input_ids[i] + [pad_tok] * padding_length)
            p_input_mask.append(input_mask[i] + [pad_tok] * padding_length)
            p_segment_ids.append(segment_ids[i] + [pad_tok] * padding_length)
            p_label_ids.append(label_ids[i] + [pad_tok] * padding_length)
            p_label_mask.append(label_mask[i] + [pad_tok] * padding_length)
        
        # Padding regional image features
        max_img_length = max(map(len, image_ids))

        for visual_emb in image_ids:
            padding_length = max_img_length - visual_emb.shape[0]
            visual_size = visual_emb.shape[1]
            visual_emb = torch.cat([visual_emb, torch.zeros(padding_length, visual_size)], dim=0).view(1, max_img_length, visual_size)
            p_image_ids.append(visual_emb)

        p_input_ids = torch.tensor(p_input_ids, dtype=torch.long)
        p_input_mask = torch.tensor(p_input_mask, dtype=torch.long)
        p_segment_ids = torch.tensor(p_segment_ids, dtype=torch.long)
        p_image_ids = torch.cat(p_image_ids, dim=0)
        p_visual_segment_ids = torch.ones(max_img_length, dtype=torch.long)
        p_visual_position_ids = torch.zeros(max_img_length, dtype=torch.long)
        p_label_ids = torch.tensor(p_label_ids, dtype=torch.long)
        p_label_mask = torch.tensor(p_label_mask, dtype=torch.long)

        # Correct the attention mask
        batch_size = len(input_ids)
        txt_length = p_input_ids.shape[1]
        img_length = p_image_ids.shape[1]

        attention_mask = torch.zeros(batch_size, txt_length + img_length)

        for i in range(batch_size):
            unmask_txt_size = p_input_mask[i].sum()
            unmask_img_size = p_image_ids[i].shape[0]
            attention_mask[i, :unmask_txt_size + unmask_img_size] = 1

        batch_dict = {
            'input_ids': p_input_ids, 'attention_mask': attention_mask, 'token_type_ids': p_segment_ids,
            'visual_embeddings': p_image_ids, 'visual_position_ids': p_visual_position_ids, 'visual_embeddings_type': p_visual_segment_ids,
            'labels': p_label_ids, 'label_mask': p_label_mask
        } 

        return batch_dict


def process_sample(tokenizer, tokens, labels, label_map, aux_text=None):
    tokenized = []
    input_msk = []
    label_ids = []
    label_msk = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        word_tokens = tokenizer.tokenize(token)
        if len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        num_subtok = len(word_tokens) - 1

        tokenized.extend(word_tokens)
        label_ids.extend([label_map[label]] + [0] * num_subtok)
        label_msk.extend([1] + [0] * num_subtok)

    tokenized = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokenized)
    input_msk = [1] * len(input_ids)
    label_ids = [0] + label_ids + [0]
    label_msk = [0] + label_msk + [0]
    segment_ids = [0] * len(input_ids)

    if aux_text is not None:
        tokenized_aux_text = tokenizer.tokenize(aux_text) + [tokenizer.sep_token]
        aux_text_ids = tokenizer.convert_tokens_to_ids(tokenized_aux_text)

        input_ids += aux_text_ids
        input_msk += [1] * len(aux_text_ids)
        label_ids += [0] * len(aux_text_ids)
        label_msk += [0] * len(aux_text_ids)
        segment_ids += [1] * len(aux_text_ids)
    
    return (tokenized, input_ids, input_msk, segment_ids, label_ids, label_msk)

