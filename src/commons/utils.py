import json
import torch
import h5py
import signal
import numpy as np

from tabulate import tabulate
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report


def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x.strip() for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def printcr(report, classes=None, sort_by_support=False):
    headers = ['classes', 'precision', 'recall', 'f1-score', 'support']

    if classes is None:
        classes = [k for k in report.keys() if k not in {'macro avg', 'micro avg'}]

        if sort_by_support:
              classes = sorted(classes, key=lambda c: report[c]['support'], reverse=True)
        else: classes = sorted(classes)

    if 'macro avg' not in classes: classes.append('macro avg')
    if 'micro avg' not in classes: classes.append('micro avg')

    table = []
    for c in classes:
        if c == 'macro avg':
            table.append([])
        row = [c]
        for h in headers:
            if h not in report[c]:
                continue
            if h in {'precision', 'recall', 'f1-score'}:
                  row.append(report[c][h] * 100)
            else: row.append(report[c][h])
        table.append(row)
    print(tabulate(table, headers=headers, floatfmt=(".3f", ".3f", ".3f", ".3f")))
    print()


class EpochStats:
    def __init__(self):
        self.sizes = [] # number of elements per step
        self.losses = []

        self.probs = []
        self.preds = []
        self.golds = []

    def loss_step(self, loss: float, batch_size: int):
        self.losses.append(loss)
        self.sizes.append(batch_size)

    def step(self, scores, target, mask, loss):
        self.loss_step(loss, len(scores))

        probs, classes = scores.max(dim=2)

        for i in range(len(scores)):
            prob_i = probs[i][mask[i] == 1].cpu().tolist()
            pred_i = classes[i][mask[i] == 1].cpu().tolist()
            gold_i = target[i][mask[i] == 1].cpu().tolist()

            self.preds.append(pred_i) # self.preds.extend(pred_i)
            self.golds.append(gold_i) # self.golds.extend(gold_i)
            self.probs.append(prob_i) # self.probs.extend(prob_i)

    def loss(self, loss_type: str = ''):
        losses = self.losses
        return np.mean([l for l, s in zip(losses, self.sizes) for _ in range(s)]), np.min(losses), np.max(losses)

    def _map_to_labels(self, index2label):
        # Predictions should have been as nested list to separate predictions
        # Since we store the predictions across epochs during training, we need to wrap up this in a try except
        # so that it handles the flattened lists in case they are not nested. New runs will be nested
        try:
            golds = [[index2label[j] for j in i] for i in self.golds]
            preds = [[index2label[j] for j in i] for i in self.preds]
        except TypeError:
            golds = [index2label[i] for i in self.golds]
            preds = [index2label[i] for i in self.preds]
        return golds, preds

    def metrics(self, index2label: [List[str], Dict[int, str]]):
        golds, preds =self._map_to_labels(index2label)

        f1 = f1_score(golds, preds)
        p = precision_score(golds, preds)
        r = recall_score(golds, preds)

        return f1, p, r

    def get_classification_report(self, index2label: [List[str], Dict[int, str]]):
        golds, preds = self._map_to_labels(index2label)

        cr = classification_report(golds, preds, digits=5)
        return report2dict(cr)

    def print_classification_report(self, index2label: [List[str], Dict[int, str]] = None, report = None):
        assert index2label is not None or report is not None

        if report is None:
            report = self.get_classification_report(index2label)

        printcr(report)


def correct_attention_mask(input_ids, attention_mask, sep_token_id=102):
    for i in range(input_ids.shape[0]):
        non_text = False
        for j in range(input_ids.shape[1]):
            if non_text:
                attention_mask[i][j] = 0
            if input_ids[i][j] == sep_token_id:
                non_text = True
    
    return attention_mask


def read_text_as_list(text_dir, caption_dir):
    ids, tokens, labels, image_ids = [], [], [], []

    with open(text_dir, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            ids.append(data['id'])
            tokens.append(data['tokens'])
            labels.append(data['label'])
            image_ids.append(data['image_id'])
    
    if caption_dir is not None:
        caption_dict = {}
        with open(caption_dir, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                caption_dict[data['image_id']] = data['caption']
        captions = []
        for idx in image_ids:
            captions.append(caption_dict[idx])
    else:
        captions = None

    return ids, tokens, labels, image_ids, captions


def read_regional_image_features_as_list(image_ids, image_dir=None):
    if image_dir is not None:
        images, objects = [], []

        image_features = torch.load(image_dir)
        for image_id in image_ids:
            box_features, cls_boxes,  max_conf, categories = image_features[image_id]
            image_feat, cls_boxes, image_loc, categories = screen_feature(box_features, cls_boxes, max_conf, categories)
            images.append(image_feat)
            objects.append(categories)
    else:
        images, objects = None, None
    
    return images, objects


def read_global_image_features_as_list(image_ids, image_dir=None):
    if image_dir is not None:
        images, objects = [], None

        image_features = h5py.File(image_dir, 'r')
        for image_id in image_ids:
            image_feat = torch.tensor(image_features[image_id].value).unsqueeze(0)
            images.append(image_feat)
    else:
        images, objects = None, None

    return images, objects


def screen_feature(image_feat, cls_boxes, max_conf, objects):
    image_feature_cap = 10

    keep_boxes = np.arange(image_feat.shape[0])

    if image_feature_cap < keep_boxes.shape[0]:
        keep_boxes = np.arange(image_feature_cap)

    image_feat = image_feat[keep_boxes]
    cls_boxes = cls_boxes[keep_boxes]
    image_loc = image_feat.shape[0]
    objects = objects[keep_boxes]

    return image_feat, cls_boxes, image_loc, objects


def flatten(l):
    return [i for sublist in l for i in sublist]


def count_params(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])


def get_label_map(datasets):
    all_labels = flatten([flatten(datasets[dataset].labels) for dataset in datasets])
    all_labels = {l: i for i, l in enumerate(sorted(set(all_labels)))}
    return all_labels


def rescale(arr):
    return (arr - arr.min((1, 2, 3), keepdims=True))/(arr.max((1, 2, 3), keepdims=True) - arr.min((1, 2, 3), keepdims=True))


def get_dataloader(dataset, batch_size, shuffle=False):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate)
    return dloader


def input_with_timeout(prompt, timeout, default=''):
    def alarm_handler(signum, frame):
        raise Exception("Time is up!")
    try:
        # set signal handler
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds

        return input(prompt)
    except Exception as ex:
        return default
    finally:
        signal.alarm(0)  # cancel alarm

