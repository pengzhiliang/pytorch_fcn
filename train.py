#-*-coding:utf-8-*-
'''
Created on Oct 24,2018

@author: pengzhiliang
'''
from __future__ import print_function
import os
import sys
import yaml
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from models import get_model
from loss import get_loss_function
from loader import get_loader 
from metrics import runningScore, averageMeter
from augmentations import get_composed_augmentations
from schedulers import get_scheduler
from optimizers import get_optimizer



def train(cfg, writer):
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
 
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            print("=====>",
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            print("=====>","No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()
            
            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()


                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())


                ptint("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k,':',v)

                for k, v in class_iou.items():
                    print('{}: {}'.format(k, v))

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join('./checkpoint',
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="config.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    print("=====> Let the games begin")

    train(cfg, writer)
