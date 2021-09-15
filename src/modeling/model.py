import torch
import torch.nn as nn

import src.commons.utils as utils

from src.modeling.layers import InferenceLayer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertModel



############################# BERT ###########################################

class NERModelBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bert = BertModel.from_pretrained(config.model_name_or_path,
                                              output_attentions=config.output_attentions,
                                              output_hidden_states=config.output_hidden_states)


    def forward_bert(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        return sequence_output
    

    def ner_loss(self, classifier, sequence_output, labels, attention_mask=None):
        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=labels.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(labels.shape, device=labels.device)

            loss, logits = classifier(sequence_output, labels, attention_mask)

        return loss, logits

    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                visual_embeddings=None, visual_position_ids=None, visual_embeddings_type=None,
                labels=None, label_mask=None, wrap_scalars=False):
        raise NotImplementedError('The NERModelBase class should never execute forward')


class NERModel(NERModelBase):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = InferenceLayer(config.hidden_size, config.num_labels, use_crf=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                visual_embeddings=None, visual_position_ids=None, visual_embeddings_type=None,
                labels=None, label_mask=None, wrap_scalars=False):

        sequence_output = self.forward_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss, logits = self.ner_loss(self.classifier, sequence_output, labels, attention_mask)

        if wrap_scalars:
            loss = loss.unsqueeze(0)

        return loss, logits


class NERWithCaption(NERModelBase):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = InferenceLayer(config.hidden_size, config.num_labels, use_crf=True)


    def multimodal_fusion(self, input_ids, attention_mask=None, token_type_ids=None, visual_embeddings=None):
        sequence_output = self.forward_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return sequence_output


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                visual_embeddings=None, visual_position_ids=None, visual_embeddings_type=None,
                labels=None, label_mask=None, next_sentence_label=None, wrap_scalars=False):

        multimodal_output = self.multimodal_fusion(input_ids, attention_mask, token_type_ids, visual_embeddings)

        # If captions exist, remove when computing loss
        attention_mask = utils.correct_attention_mask(input_ids, attention_mask)
        
        loss, logits = self.ner_loss(self.classifier, multimodal_output, labels, attention_mask)

        if wrap_scalars:
            loss = loss.unsqueeze(0)

        return loss, logits


############################# VisualBERT #####################################

class BertEmbeddingsWithVisualEmbedding(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsWithVisualEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #### Below are specific for encoding visual features

        # Segment and position embedding for image features
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        if hasattr(config, 'visual_embedding_dim'):
            self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, 
                visual_embeddings=None, visual_embeddings_type=None, visual_position_ids=None):
        '''
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        '''

        input_shape = input_ids.size()
        seq_length = input_ids.size(1)
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if visual_embeddings is not None:
            visual_embeddings = self.projection(visual_embeddings)

            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            # visual_position_ids = torch.zeros(*visual_embeddings.size()[:-1], dtype = torch.long)#.cuda()
            position_embeddings_visual = self.position_embeddings_visual(visual_position_ids)

            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concate the two:
            embeddings = torch.cat((embeddings, v_embeddings), dim = 1) # concat the visual embeddings after the attentions
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddingsWithVisualEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                visual_embeddings=None, visual_embeddings_type=None, visual_position_ids=None):

        input_shape = input_ids.size()
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_extended_attention_mask = None
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,  position_ids=position_ids, 
                                        visual_embeddings=visual_embeddings, visual_embeddings_type=visual_embeddings_type, visual_position_ids=visual_position_ids)

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
                                    encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, output_attentions=self.config.output_attentions)
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class MNERModelBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bert = VisualBertModel(config)

        self._load_pretrained_weights(config.ckpt_path)
    
    def _load_pretrained_weights(self, ckpt_path):
        old_state = self.bert.state_dict()
        new_state = torch.load(ckpt_path)
        for pname in old_state.keys():
            pname_norm = 'bert.bert.' + pname
            if pname_norm in new_state:
                old_state[pname] = new_state[pname_norm]
            else:
                print("[LOG] Missing: {} ({})".format(pname, pname_norm))
            
        self.bert.load_state_dict(old_state)

    def forward_bert(self, input_ids, attention_mask=None, token_type_ids=None, 
                    visual_embeddings=None, visual_embeddings_type=None, visual_position_ids=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,
                            visual_embeddings=visual_embeddings, visual_embeddings_type=visual_embeddings_type, visual_position_ids=visual_position_ids)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        return sequence_output

    def ner_loss(self, classifier, sequence_output, labels, attention_mask=None):
        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=labels.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(labels.shape, device=labels.device)

            loss, logits = classifier(sequence_output, labels, attention_mask)

        return loss, logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                visual_embeddings=None, visual_embeddings_type=None, visual_position_ids=None,
                labels=None, label_mask=None, wrap_scalars=False):
        raise NotImplementedError('The NERModelBase class should never execute forward')


class MNERModel(MNERModelBase):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = InferenceLayer(config.hidden_size, config.num_labels, use_crf=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                visual_embeddings=None, visual_embeddings_type=None, visual_position_ids=None,
                labels=None, label_mask=None, wrap_scalars=False):

        sequence_output = self.forward_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                            visual_embeddings=visual_embeddings, visual_embeddings_type=visual_embeddings_type, visual_position_ids=visual_position_ids)
        
        attention_mask = utils.correct_attention_mask(input_ids, attention_mask)

        # Remove objects
        sequence_output = sequence_output[:, :labels.shape[1], :]
        attention_mask = attention_mask[:, :labels.shape[1]]

        loss, logits = self.ner_loss(self.classifier, sequence_output, labels, attention_mask)

        if wrap_scalars:
            loss = loss.unsqueeze(0)

        return loss, logits

