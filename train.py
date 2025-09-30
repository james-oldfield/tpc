import copy
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from utils import get_current_order
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.nn as nn

def train(classifier, train_loader, val_loader, num_epochs, train_mode, min_order, max_order, metric, num_seeds=1, lr=1e-4, weight_decay=0.1, d_prob=0.0, device=torch.device('cuda'), **kwargs):
    cls_all = []
    num_params_all = []

    if train_mode == 'progressive':
        num_epochs = num_epochs * (max_order - 1)

    for seed in tqdm(range(num_seeds), desc='Training seeds'):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cls = classifier(train_mode=train_mode,d_prob=d_prob).to(device)

        optimizer = torch.optim.AdamW(cls.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        num_params = sum([x.numel() for x in list(cls.parameters())])
        if seed == 0:
            print(f'Num params: {format(num_params, ',')}')
            print(f'Training mode: {train_mode}')
        
        prev_order = min_order
        
        # training
        for epoch in range(1, num_epochs + 1):
            cls.train()
            running_loss = 0.0
            train_preds, train_labels = [], []

            for it, (xb_cpu, yb_cpu) in enumerate(train_loader):
                if train_mode == 'progressive':
                    current_order = get_current_order(epoch, num_epochs, min_order=min_order, max_order=max_order)
                    if current_order > prev_order:
                        # freeze previous poly terms (or layers)
                        for name, param in cls.named_parameters():
                            for o in range(0, current_order):
                                if f"HO.{(o-2)}." in name or name == f"alpha.{(o-2)}" or "W" in name:
                                    param.requires_grad = False
                        prev_order = current_order

                        # reset the scheduler to the initial learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        tqdm.write(f"-----------> Now using polynomial order: {current_order}/{max_order}")
                elif train_mode == 'random':
                    current_order = random.randint(min_order, max_order)
                elif train_mode == 'max' or train_mode == 'train_all':
                    current_order = max_order
                else:
                    print(f'Unknown train mode: {train_mode}')
                    raise

                cls.current_order = current_order
                
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)

                optimizer.zero_grad()

                logits = cls(xb,  test_time_order=current_order)
                if train_mode == 'train_all':
                    loss = torch.mean(torch.stack([criterion(out.squeeze(), yb) for out in logits]))
                else:
                    # else, just use the last output
                    loss = criterion(logits[-1].squeeze(-1), yb)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cls.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.detach() * xb.size(0)

                train_preds.append(logits[-1].squeeze(-1).cpu())
                train_labels.append(yb.cpu())
        
            # epoch‐level train metrics
            train_preds  = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            train_acc = metric(train_preds, train_labels.round().long())
            epoch_loss   = running_loss.item() / len(train_loader.dataset)
        
            # validation
            cls.eval()
            val_preds, val_labels = [], []
        
            running_val_loss = 0.0
            with torch.no_grad():
                for xb_cpu, yb_cpu in val_loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
        
                    # use max order for val
                    if train_mode != 'progressive':
                        current_order = max_order
                    logits = cls(xb, test_time_order=current_order)
                    logits = logits[-1].squeeze(-1)
                    running_val_loss += criterion(logits, yb).item() * xb.size(0)

                    val_preds.append(logits.cpu())
                    val_labels.append(yb.cpu())

            epoch_loss_val = running_val_loss / len(val_loader.dataset)
        
            val_preds  = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            val_acc = metric(val_preds, val_labels.long())

            scheduler.step(epoch_loss_val)

            current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(
            f"Epoch {epoch}/{num_epochs}"
            f" - val_acc: {val_acc:.4f}"
            f" - train_acc: {train_acc:.4f}"
            f" - train_loss: {epoch_loss:.4f}"
            f" - current order: {current_order}"
            f" - current LR: {current_lr:.6f}"
        )

        cls_all.append(cls)
        num_params_all.append(num_params)

    return cls_all, num_params, epoch, val_acc


# simply a grid search over hyperparameters w/ validation set
def hyperparam_sweep(lrs, wds, douts, classifier, train_loader, val_loader, num_epochs, train_mode, min_order, max_order, metric, device=torch.device('cuda')):
    vals = []
    configs = []
    for lr in lrs:
        for wd in wds:
            for d_prob in douts:
                _, _, epoch, val_acc = train(
                    classifier, train_loader, val_loader, num_epochs, train_mode, min_order, max_order, metric, num_seeds=1,
                    lr=lr, weight_decay=wd, d_prob=d_prob, device=device)

                vals.append(val_acc.item())
                configs.append((lr, wd, d_prob, epoch))
        
    return configs[np.argmax(vals)], np.max(vals)


def ee_sweep(cls_all, train_loader, val_loader, test_loader, x_vals, metric, device=torch.device('cuda')):
    """
    Evaluate the classifiers with a range of test-time orders / partial early-exit layers.
    """
    y_vals_test = []
    y_vals_val = []
    y_vals_train = []

    for cls in cls_all:
        y_vals_test_local = []
        y_vals_val_local = []
        y_vals_train_local = []
        cls.eval()

        def evaluate_order(o, loader):
            bar = tqdm( loader, leave=False, unit="batch" )
            preds, labels = [], []
            with torch.no_grad():
                for xb_cpu, yb_cpu in bar:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
                    logits = cls(xb, test_time_order=o)[-1].squeeze(-1)

                    preds.append(logits.cpu())
                    labels.append(yb.cpu())

            return metric(torch.cat(preds), torch.cat(labels).long())

        for o in x_vals:
            y_vals_train_local.append(evaluate_order(o, train_loader))
            y_vals_val_local.append(evaluate_order(o, val_loader))
            y_vals_test_local.append(evaluate_order(o, test_loader))

        y_vals_train.append(y_vals_train_local)
        y_vals_val.append(y_vals_val_local)
        y_vals_test.append(y_vals_test_local)

    tr = np.array(y_vals_train)
    val = np.array(y_vals_val)
    te = np.array(y_vals_test)

    print(f'Metric: {metric}')

    return (tr, val, te, x_vals)


                    
def cascade(cls, xb, tau, temperatures=None, min_idx=1, max_casc_order=None):
    B     = xb.size(0)
    logits = torch.zeros(B, device=xb.device)
    exit_deg  = torch.full((B,), -1, dtype=torch.long, device=xb.device)
    active_idx = torch.arange(B, device=xb.device)

    cls.eval()

    if temperatures is None:
        temperatures = np.ones((max_casc_order, 2), dtype=np.float32)
        # disable bias scaling
        temperatures[:, 1] = 0.0

    ## -- degree 1
    f1 = cls(xb, test_time_order=min_idx)[-1]
    logits = f1.squeeze(-1)

    safe   = torch.sigmoid(logits[active_idx]/temperatures[min_idx-1][0]+temperatures[min_idx-1][1]) < tau[0]
    unsafe = torch.sigmoid(logits[active_idx]/temperatures[min_idx-1][0]+temperatures[min_idx-1][1]) > tau[1] 
    
    exit_deg[active_idx[safe | unsafe]] = 1 # all elements go to at least deg 1
    active_idx = active_idx[~(safe | unsafe)] 

    for o in range(min_idx+1, max_casc_order+1 or cls.max_order + 1):

        if active_idx.numel() > 0:
            f = cls(xb[active_idx], test_time_order=o)[-1]

            logits[active_idx] = f.squeeze(-1)
            safe   = torch.sigmoid(logits[active_idx]/temperatures[o-1][0]+temperatures[o-1][1]) < tau[0]
            unsafe = torch.sigmoid(logits[active_idx]/temperatures[o-1][0]+temperatures[o-1][1]) > tau[1]

            # these elements reach at least degree o
            exit_deg[active_idx] = o
                
            # shrink the active set based on those not yet classified
            active_idx = active_idx[~(safe | unsafe)]
    return logits, exit_deg