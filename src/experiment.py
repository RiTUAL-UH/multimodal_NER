import os
import torch

import src.commons.utils as utils

from transformers import BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from src.data.dataset import NERDataset, NERDatasetWithGloablImageFeatures, NERDatasetWithRegionalImageFeatures
from src.modeling.model import NERModel, NERWithCaption, MNERModel
from src.modeling.train import *


def get_dataloaders(args, tokenizer):
    label_scheme = args.data.label_scheme

    if args.data.dataset_class == 'ner':
        datasets = {
            'train': NERDataset(args.data.text.train, args.data.image.train, args.data.caption.train, tokenizer, label_scheme),
            'dev'  : NERDataset(args.data.text.dev, args.data.image.dev, args.data.caption.dev, tokenizer, label_scheme),
            'test' : NERDataset(args.data.text.test, args.data.image.test, args.data.caption.test, tokenizer, label_scheme)
        }
    elif args.data.dataset_class == 'ner_with_global_image_features':
        datasets = {
            'train': NERDatasetWithGloablImageFeatures(args.data.text.train, args.data.image.train, args.data.caption.train, tokenizer, label_scheme),
            'dev'  : NERDatasetWithGloablImageFeatures(args.data.text.dev, args.data.image.dev, args.data.caption.dev, tokenizer, label_scheme),
            'test' : NERDatasetWithGloablImageFeatures(args.data.text.test, args.data.image.test, args.data.caption.test, tokenizer, label_scheme)
        }
    elif args.data.dataset_class == 'ner_with_regional_image_features':
        datasets = {
            'train': NERDatasetWithRegionalImageFeatures(args.data.text.train, args.data.image.train, args.data.caption.train, tokenizer, label_scheme),
            'dev'  : NERDatasetWithRegionalImageFeatures(args.data.text.dev, args.data.image.dev, args.data.caption.dev, tokenizer, label_scheme),
            'test' : NERDatasetWithRegionalImageFeatures(args.data.text.test, args.data.image.test, args.data.caption.test, tokenizer, label_scheme)
        }
    else:
        raise NotImplementedError("Unexpected dataset class")

    print("[LOG] Train data size: {:,}".format(len(datasets['train'])))
    print("[LOG]   Dev data size: {:,}".format(len(datasets['dev'])))
    print("[LOG]  Test data size: {:,}".format(len(datasets['test'])))

    train_batch_size = args.training.per_gpu_train_batch_size * max(1, args.optim.n_gpu)
    eval_batch_size = args.training.per_gpu_eval_batch_size * max(1, args.optim.n_gpu)
    dataloaders = {
        'train': utils.get_dataloader(datasets['train'], train_batch_size, shuffle=True),
        'dev'  : utils.get_dataloader(datasets['dev'], eval_batch_size, shuffle=False),
        'test' : utils.get_dataloader(datasets['test'], eval_batch_size, shuffle=False)
    }

    return dataloaders


def prepare_model_config(args):
    config = BertConfig.from_pretrained(args.model.model_name_or_path)
    config.model_name_or_path = args.model.model_name_or_path
    config.num_labels = len(args.data.label_scheme)
    config.output_attentions = args.model.output_attentions
    config.output_hidden_states = args.model.output_hidden_states
    config.visual_embedding_dim = args.data.image.embedding_dim
    config.visual_embedding_size = args.data.image.size
    config.ckpt_path = args.model.pretrained_weights

    return config


def get_model_class(model_name):
    if model_name == 'ner':
        model_class = NERModel
    
    elif model_name == 'ner_with_caption':
        model_class = NERWithCaption
    
    elif model_name == 'mner':
        model_class = MNERModel

    else:
        raise NotImplementedError

    return model_class


def get_optimizer_and_scheduler(args, model, num_training_sampels):
    oargs = args.optim

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (num_training_sampels // oargs.gradient_accumulation_steps) + 1
    else:
        t_total = num_training_sampels // oargs.gradient_accumulation_steps * args.training.epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": oargs.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=oargs.learning_rate,
                      betas=(oargs.beta_1, oargs.beta_2),
                      eps=oargs.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)

    return optimizer, scheduler


def print_stats(stats, label_scheme):
    for i in range(len(stats['train'])):
        print(f"Epoch {i + 1} -", end=" ")
        for split in ['train', 'dev']:
            epoch_stats = stats[split][i]
            f1, _, _ = epoch_stats.metrics(label_scheme)
            loss = sum(epoch_stats.losses) / len(epoch_stats.losses)
            print(f"[{split.upper()}] F1: {f1 * 100:.3f} Loss: {loss:.5f}", end=' ')
        print()
    print()


def main(args):

    tokenizer = BertTokenizer.from_pretrained(args.model.model_name_or_path, do_lower_case=args.model.do_lower_case)

    dataloaders = get_dataloaders(args, tokenizer)

    config = prepare_model_config(args)

    model = get_model_class(args.model.name)(config)

    print("[LOG] " + "=" * 40)
    print("[LOG] Parameter count: {}".format(utils.count_params(model)))
    print("[LOG] " + "=" * 40)
    print()

    if args.experiment.do_training == True:
        confirm = 'y'
        if os.path.exists(args.experiment.checkpoint_dir):
            confirm = input('A checkpoint was detected. Do you really want to train again and override the model? [y/n]: ').strip()
            
        if confirm != 'y':
            print("[LOG] Skip training")
            stats = {
                'train': torch.load(os.path.join(args.experiment.output_dir, 'train_preds_across_epochs.bin')),
                'dev': torch.load(os.path.join(args.experiment.output_dir, 'dev_preds_across_epochs.bin'))
            }
            print_stats(stats, args.data.label_scheme)
        else:
            model.to(args.optim.device)
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(dataloaders['train']))
            stats, f1, global_step = train(args, model, dataloaders, optimizer, scheduler)
            
            print_stats(stats, args.data.label_scheme)
            print(f"[LOG] Best dev F1: {f1:.5f}")
            print(f"[LOG] Best global step: {global_step}")
            print()
        
    if utils.input_with_timeout("Do you want to evaluate the model? [y/n]:", 15, "y").strip() == 'y':
        # Perform evaluation over the dev and test sets with the best checkpoint
        print(f"[LOG] Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
        model = get_model_class(args.model.name).from_pretrained(args.experiment.checkpoint_dir)
        model.to(args.optim.device)

        for split in dataloaders.keys():
            if split == 'train':
                continue

            stats = predict(args, model, dataloaders[split])
            torch.save(stats, os.path.join(args.experiment.output_dir, f'{split}_best_preds.bin'))

            # f1, prec, recall = stats.metrics(args.data.label_scheme)
            loss, _, _ = stats.loss()

            report = stats.get_classification_report(args.data.label_scheme)
            classes = sorted(set([label[2:] for label in args.data.label_scheme if label != 'O']))

            print(f"\n********** {split.upper()} RESULTS **********\n")
            print('\t'.join(["Loss"] + classes + ["F1"]), end='\n')
            print('\t'.join([f"{l:.4f}" for l in [loss]]), end='\t')
            f1_scores = []
            for c in classes + ["micro avg"]:
                if 'f1-score' in report[c].keys():
                    f1_scores.append(report[c]['f1-score'])
                else:
                    f1_scores.append(0)
            print('\t'.join([f"{score * 100:.3f}" for score in f1_scores]))
            print()
            
            if utils.input_with_timeout("Print class-level results? [y/n]:", 5, "n").strip() == 'y':
                stats.print_classification_report(report=report)
        print()    


if __name__ == '__main__':
    main()

