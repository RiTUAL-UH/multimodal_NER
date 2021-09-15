import torch
import torch.nn as nn

from allennlp.modules import ConditionalRandomField


class InferenceLayer(nn.Module):
    def __init__(self, input_dim, n_classes, use_crf):
        super(InferenceLayer, self).__init__()

        self.use_crf = use_crf
        self.input_dim = input_dim
        self.output_dim = n_classes

        self.proj = nn.Linear(input_dim, n_classes)

        if self.use_crf:
            self.crf = ConditionalRandomField(n_classes, constraints=None, include_start_end_transitions=True)
            # self.crf = CRF(n_classes, batch_first=True)
        else:
            self.xent = nn.CrossEntropyLoss(reduction='mean')

    def crf_forward(self, logits, target, mask):
        mask = mask.long()
        # best_paths = self.crf.viterbi_tags(logits, mask)
        # tags, viterbi_scores = zip(*best_paths)
        loss = -self.crf.forward(logits, target, mask)  # neg log-likelihood loss
        loss = loss / torch.sum(mask)
        
        return loss, logits

    def fc_forward(self, logits, target, mask):
        mask = mask.long()
        mask = mask.view(-1) == 1

        logits_ = logits.view(-1, logits.size(-1))
        target_ = target.view(-1)

        loss = self.xent(logits_[mask], target_[mask])

        return loss, logits

    def forward(self, vectors, targets, mask):
        logits = self.proj(vectors)

        if self.use_crf:
            loss, logits = self.crf_forward(logits, targets, mask)
        else:
            loss, logits = self.fc_forward(logits, targets, mask)

        return loss, logits

