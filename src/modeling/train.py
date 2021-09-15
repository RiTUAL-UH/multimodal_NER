
import os
import torch
import gc

import src.commons.utils as utils

from tqdm import tqdm, trange


def decode_labels(label_map, encoded_labels):
    index_to_label = {index: label for label, index in label_map.items()}

    for i in range(len(encoded_labels)):
        for j in range(len(encoded_labels[i])):
            encoded_labels[i][j] = index_to_label[encoded_labels[i][j]]

    return encoded_labels


def track_best_model(args, model, dev_stats, best_f1, best_step, global_step):
    curr_f1, _, _ = dev_stats.metrics(args.data.label_scheme)
    if best_f1 >= curr_f1:
        return best_f1, best_step

    # Save model checkpoint
    os.makedirs(args.experiment.checkpoint_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.experiment.checkpoint_dir)
    meta = {
        'args': args,
        'f1': curr_f1,
        'global_step': global_step
    }
    torch.save(meta, os.path.join(args.experiment.checkpoint_dir, "training_meta.bin"))
    return curr_f1, global_step


def predict(args, model, dataloaders):
    model.eval()
    stats = utils.EpochStats()

    oargs = args.optim

    # multi-gpu evaluate
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for batch_dict in tqdm(dataloaders):
        for field in batch_dict:
            if batch_dict[field] is not None:
                batch_dict[field] = batch_dict[field].to(oargs.device)

        outputs = model(**batch_dict, wrap_scalars=oargs.n_gpu > 1)
        loss = outputs[0]

        if oargs.n_gpu > 1:
            # There is one parallel loss per device
            loss = loss.mean()

        stats.step(scores=outputs[1], target=batch_dict['labels'], mask=batch_dict['label_mask'], loss=loss.item())

    return stats


def train(args, model, dataloaders, optimizer, scheduler):
    oargs = args.optim

    # multi-gpu training
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    best_f1, best_step = 0., 0
    global_step = 0
    stats = {'train': [], 'dev': []}
    
    epoch_desc = "Epochs (Dev F1: {:.5f} at step {})"
    epoch_iterator = trange(int(args.training.epochs), desc=epoch_desc.format(best_f1, best_step))

    for _ in epoch_iterator:
        epoch_iterator.set_description(epoch_desc.format(best_f1, best_step), refresh=True)

        for split in ['train', 'dev']:
            epoch_stats = utils.EpochStats()
            batch_iterator = tqdm(dataloaders[split], desc=f"{split.title()} iteration")
            # ====================================================================
            for step, batch_dict in enumerate(batch_iterator):
                if split == 'train':
                    model.train()
                    model.zero_grad()
                else:
                    model.eval()

                for field in batch_dict.keys():
                    if batch_dict[field] is not None:
                        batch_dict[field] = batch_dict[field].to(oargs.device)
                        
                outputs = model(**batch_dict, wrap_scalars=oargs.n_gpu > 1)
                loss = outputs[0]

                if oargs.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
                if oargs.gradient_accumulation_steps > 1:
                    loss = loss / oargs.gradient_accumulation_steps

                epoch_stats.step(scores=outputs[1], target=batch_dict['labels'], mask=batch_dict['label_mask'], loss=loss.item())

                if split == 'train':
                    loss.backward()

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
                
                if oargs.max_steps > 0 and global_step > oargs.max_steps:
                    batch_iterator.close()
                    break

                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    _ = gc.collect()

            # ====================================================================
            stats[split].append(epoch_stats)

            if split == 'dev':
                best_f1, best_step = track_best_model(args, model, epoch_stats, best_f1, best_step, global_step)

        os.makedirs(args.experiment.output_dir, exist_ok=True)
        torch.save(stats['train'], os.path.join(args.experiment.output_dir, 'train_preds_across_epochs.bin'))
        torch.save(stats['dev'], os.path.join(args.experiment.output_dir, 'dev_preds_across_epochs.bin'))

        if oargs.max_steps > 0 and global_step > oargs.max_steps:
            epoch_iterator.close()
            break

    return stats, best_f1, best_step

