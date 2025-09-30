from torch.utils.data import TensorDataset, DataLoader
from functools import partial
from train import train, ee_sweep, hyperparam_sweep
from model import PolyProbe, MLP, LinearProbe, BilinearProbe, EEMLP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torchmetrics.classification import F1Score
from tqdm import tqdm

import torch
import numpy as np
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a monitor for Llama WildGuard")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max_order", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--layer_idx", type=int, default=40)
    parser.add_argument("--pool_type", type=str, default="mean")
    parser.add_argument("--train_mode", type=str, default="progressive")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--dataset", type=str, default="WildGuard", help="Dataset to use for training", choices=["WildGuard", "BeaverTrails"])
    return parser.parse_args()

args = parse_args()

sublabels_names = ['benign', 'causing_material_harm_by_disseminating_misinformation', 'copyright_violations', 'cyberattack', 'defamation_encouraging_unethical_or_unsafe_actions', 'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns', 'fraud_assisting_illegal_activities', 'mental_health_over-reliance_crisis', 'others', 'private_information_individual', 'sensitive_information_organization_government', 'sexual_content', 'social_stereotypes_and_unfair_discrimination', 'toxic_language_hate_speech', 'violence_and_physical_harm'] \
    if args.dataset == "WildGuard" else ['', 'Controlled/Regulated Substances', 'Copyright/Trademark/Plagiarism', 'Criminal Planning/Confessions', 'Fraud/Deception', 'Guns and Illegal Weapons', 'Harassment', 'Hate/Identity Hate', 'High Risk Gov Decision Making', 'Illegal Activity', 'Immoral/Unethical', 'Malware', 'Manipulation', 'Needs Caution', 'Other', 'PII/Privacy', 'Political/Misinformation/Conspiracy', 'Profanity', 'Sexual', 'Sexual (minor)', 'Suicide and Self Harm', 'Threat', 'Unauthorized Advice', 'Violence']

dataset = args.dataset
max_epochs = args.max_epochs
max_order = args.max_order
rank = args.rank
train_mode = args.train_mode
layer_idx = args.layer_idx
pool_type = args.pool_type

# number of models to train and average results over
num_seeds = 5
batch_size = args.batch_size

# hyperparameter sweeps
lrs_sweep = [1e-3, 5e-4, 1e-4]
wds_sweep = [1.00, 0.1, 0.01]
douts_sweep = [0.0, 0.2, 0.5]

# print all configs
print(f"Dataset: {dataset}")
print(f"Model name: {args.model_name}")
print(f"Training mode: {train_mode}")
print(f"Layer index: {args.layer_idx}")
print(f"Pooling type: {args.pool_type}")
print(f"Rank: {rank}")
print(f"Max order: {max_order}")

root_dir = "/data/scratch/acw663/poly-monitor/activation-cache/"
model_name = args.model_name

if dataset == 'BeaverTrails': pool_type += '-330k_'
else: pool_type += '-'

X = np.load(open(f'{root_dir}/{dataset}/{model_name.split('/')[-1]}-L{layer_idx}-{pool_type}train-X.npy', 'rb'))
y = np.load(open(f'{root_dir}/{dataset}/{model_name.split('/')[-1]}-{pool_type}train-y.npy', 'rb'))
X_test = np.load(open(f'{root_dir}/{dataset}/{model_name.split('/')[-1]}-L{layer_idx}-{pool_type}test-X.npy', 'rb'))
y_test = np.load(open(f'{root_dir}/{dataset}/{model_name.split('/')[-1]}-{pool_type}test-y.npy', 'rb'))

np.random.seed(42)
rnd_train_idx = np.random.choice(len(y), size=int(0.8 * len(y)), replace=False)
rnd_val_idx = np.setdiff1d(np.arange(len(y)), rnd_train_idx)
X_train, X_val = X[rnd_train_idx], X[rnd_val_idx]
y_train, y_val = y[rnd_train_idx], y[rnd_val_idx]

if dataset == 'WildGuard':
    output_dir = f'./results/numbers/wildguard/{model_name.split("/")[-1]}-L{layer_idx}-{pool_type}-{train_mode}-maxorder-{max_order}_rank-{rank}'
else:
    output_dir = f'./results/numbers/beaver/{model_name.split("/")[-1]}-L{layer_idx}-{pool_type}-{train_mode}-maxorder-{max_order}_rank-{rank}'
os.makedirs(output_dir, exist_ok=True)

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_val = scalar.transform(X_val)
X_test = scalar.transform(X_test)
 
clfs = []; vals = []
for C in tqdm([100, 10, 1, 0.1, 0.01, 0.001]):
    c = LogisticRegression(max_iter=500, C=C)
    c.fit(X_train, y_train)
    y_val_hat = c.predict(X_val)
    
    clfs.append(c)
    vals.append(f1_score(y_val, y_val_hat))

# set the clf to the one with highest val f1 score
clf = clfs[np.argmax(vals)]
print(f"f1 val = {np.max(vals):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.from_numpy(X_train).float()
X_val_t = torch.from_numpy(X_val).float()
X_test_t  = torch.from_numpy(X_test).float()
y_train_t = torch.from_numpy(y_train).float()
y_val_t = torch.from_numpy(y_val).float()
y_test_t  = torch.from_numpy(y_test).float()

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, pin_memory=True)

metric = F1Score('binary').to('cpu')
metric_name = 'F1'
out_features = 1

linear_init = [clf.intercept_, clf.coef_] # init with the linear probe weights
start_idx = 2 # if we have a linear init, we start at order 2, otherwise at order 1

# populate with generic params
train = partial(train, train_loader=train_loader, val_loader=val_loader, metric=metric)
ee_sweep = partial(ee_sweep, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, metric=metric)

########################
######################## POLY
########################
ranks = [rank]*(max_order-1)

print('######################')
print('##### Poly probe #####')
print('--------- Tuning on validation set')
model_train = partial(PolyProbe, in_features=X_train.shape[-1], ranks=ranks, out_features=out_features, max_order=max_order, linear_init=linear_init)
[lr, wd, d_prob, best_epoch], val_best = hyperparam_sweep(lrs_sweep, wds_sweep, douts_sweep, model_train, train_loader, val_loader, max_epochs, train_mode=train_mode, min_order=start_idx, max_order=max_order, metric=metric)

print('--------- Training with best hyperparameters')
poly_classifiers, poly_num_params, _, _ = train(model_train, num_epochs=max_epochs, lr=lr, weight_decay=wd, d_prob=d_prob, train_mode=train_mode, min_order=start_idx, max_order=max_order, num_seeds=num_seeds)

print('--------- Evaluating dynamic performance')
x_vals = list(range(1, max_order+1))
y_vals_train, y_vals_val, y_vals_test, xvals = ee_sweep(cls_all=poly_classifiers, x_vals=x_vals)

############### CP
# count the alphas as well
order_new_params = [poly_classifiers[0].W[1].numel()] + [sum([param.numel() + poly_classifiers[0].lam[o].numel() for param in poly_classifiers[0].HO[o]]) for o in range(0, max_order-1)] # bias term + other params
order_cumulative_params = [poly_classifiers[0].W[0].numel() + sum(order_new_params[:(o+1)]) for o in range(len(order_new_params))]

# save yvals to file
pickle.dump({'x_vals': x_vals, 'y_vals_train': y_vals_train, 'y_vals_val': y_vals_val, 'y_vals_test': y_vals_test,
             'lr': lr, 'wd': wd, 'd_prob': d_prob, 'best_epoch': best_epoch, 'val_best': val_best,
             'params': order_new_params, 'cumulative_params': order_cumulative_params}, open(os.path.join(output_dir, 'poly-vals.pkl'), 'wb'))

########################
######################## bilinear
########################
print('##### Bilinear probe #####')
model_train = partial(BilinearProbe, in_features=X_train.shape[-1], rank=rank, out_features=out_features, symmetric=True)
[lr, wd, _, best_epoch], val_best = hyperparam_sweep(lrs_sweep, wds_sweep, [0], model_train, train_loader, val_loader, max_epochs, train_mode='max', min_order=start_idx, max_order=max_order, metric=metric)

bi_classifiers, bi_num_params, _, _ = train(model_train, lr=lr, weight_decay=wd, d_prob=0.0, num_epochs=max_epochs, train_mode='max', min_order=start_idx, max_order=max_order, num_seeds=num_seeds)

x_vals = list(range(1, 2))
y_vals_bi_train, y_vals_bi_val, y_vals_bi_test, xvals = ee_sweep(bi_classifiers, x_vals=x_vals)

pickle.dump({'x_vals': x_vals, 'y_vals_train': y_vals_bi_train, 'y_vals_val': y_vals_bi_val, 'y_vals_test': y_vals_bi_test,
             'lr': lr, 'wd': wd, 'd_prob': 0, 'best_epoch': best_epoch, 'val_best': val_best,
             'params': bi_num_params, 'cumulative_params': bi_num_params}, open(os.path.join(output_dir, 'bi-vals.pkl'), 'wb'))

########################
######################## linear
########################
print('##### Linear probe #####')
linear_classifier = LinearProbe(input_dim=X_train.shape[-1], output_dim=1).to(device)

linear_classifier.net.weight.data = torch.from_numpy(clf.coef_).float().to(device)
linear_classifier.net.bias.data = torch.from_numpy(clf.intercept_).float().to(device)

x_vals = list(range(1, 2))
y_vals_linear_train, y_vals_linear_val, y_vals_linear_test, xvals = ee_sweep([linear_classifier], x_vals=x_vals)

linear_num_params = [linear_classifier.net.weight.numel() + linear_classifier.net.bias.numel()]

pickle.dump({'x_vals': xvals, 'y_vals_train': y_vals_linear_train, 'y_vals_val': y_vals_linear_val, 'y_vals_test': y_vals_linear_test,
             'lr': 0, 'wd': 0, 'd_prob': 0, 'best_epoch': 0, 'val_best': 0,
             'params': linear_num_params, 'cumulative_params': linear_num_params}, open(os.path.join(output_dir, 'linear-vals.pkl'), 'wb'))

########################
######################## Static MLPs
########################
print('##### Static MLPs (takes a while -- training a new separate MLP per param count) #####')
from utils import compute_hidden_size
hidden_dims = [1] + [compute_hidden_size(params, input_dim=X_train.shape[1], output_dim=out_features) for params in order_cumulative_params[1:]]
x_vals = list(range(1, 2))

y_vals_mlp_train = []
y_vals_mlp_val = []
y_vals_mlp_test = []
mlp_hyperparams = []
mlp_params = []

for hi, h in enumerate(hidden_dims):
    ### optimize hyper params
    if hi == 0: model_train = partial(LinearProbe, input_dim=X_train.shape[-1], output_dim=out_features)
    else: model_train = partial(MLP, input_dim=X_train.shape[-1], hidden_dim=h, output_dim=out_features)
    [lr, wd, d_prob, best_epoch], val_best = hyperparam_sweep(lrs_sweep, wds_sweep, douts_sweep, model_train, train_loader, val_loader, max_epochs, train_mode='max', min_order=1, max_order=max_order, metric=metric)
    
    # train
    mlp_classifiers, mlp_num_params, _, _ = train(model_train, lr=lr, weight_decay=wd, d_prob=d_prob, num_epochs=max_epochs, train_mode='max', min_order=1, max_order=max_order, num_seeds=num_seeds)

    y_vals_mlp_train_local, y_vals_mlp_val_local, y_vals_mlp_test_local, _ = ee_sweep(mlp_classifiers, x_vals=x_vals)
    
    y_vals_mlp_train += [y_vals_mlp_train_local]
    y_vals_mlp_val += [y_vals_mlp_val_local]
    y_vals_mlp_test += [y_vals_mlp_test_local]
    mlp_params += [sum(param.numel() for param in mlp_classifiers[0].state_dict().values())]
    mlp_hyperparams += [[lr, wd, d_prob, best_epoch, val_best]]

# save yvals to file
pickle.dump({'x_vals': x_vals, 'y_vals_train': y_vals_mlp_train, 'y_vals_val': y_vals_mlp_val, 'y_vals_test': y_vals_mlp_test,
             'lr': [hp[0] for hp in mlp_hyperparams], 'wd': [hp[1] for hp in mlp_hyperparams], 'd_prob': [hp[2] for hp in mlp_hyperparams], 'best_epoch': [hp[3] for hp in mlp_hyperparams], 'val_best': [hp[4] for hp in mlp_hyperparams],
             'params': mlp_params, 'cumulative_params': mlp_params}, open(os.path.join(output_dir, 'mlp-vals.pkl'), 'wb'))

########################
######################## EE-MLP
########################
print('##### Training the early exit MLP #####')
from utils import compute_ee_hidden_sizes
ee_hidden_dims = compute_ee_hidden_sizes(order_cumulative_params[1:], input_dim=X_train.shape[1], output_dim=out_features)

increasing_depth = 0 # turn off; each early-exit block is just one layer, but with different hidden sizes.
ee_layers = len(ee_hidden_dims)

model_train = partial(EEMLP, input_dim=X_train.shape[-1], hidden_dims=ee_hidden_dims, num_layers=ee_layers, output_dim=out_features, increasing_depth=increasing_depth)
ee_train_mode = 'train_all'
[lr, wd, d_prob, best_epoch], val_best = hyperparam_sweep(lrs_sweep, wds_sweep, douts_sweep, model_train, train_loader, val_loader, max_epochs, train_mode=ee_train_mode, min_order=1, max_order=ee_layers+1, metric=metric)

eemlp_classifiers, eemlp_num_params, _, _ = train(model_train, lr=lr, weight_decay=wd, d_prob=d_prob, num_epochs=max_epochs, train_mode=ee_train_mode, min_order=1, max_order=ee_layers+1, num_seeds=num_seeds)

eemlp_matched_x_vals = list(range(1, ee_layers+2)) # start from 0 to include linear term
y_vals_eemlpmatched_train, y_vals_eemlpmatched_val, y_vals_eemlpmatched_test, xvals = ee_sweep(eemlp_classifiers, x_vals=eemlp_matched_x_vals)
eemlp_matched_params = []
eemlp_matched_cumulative_params = []

m = eemlp_classifiers[0]
def nparams(mod):
    return sum(p.numel() for p in mod.parameters())

layer_params = [nparams(layer) for layer in m.layers]
branch_params = [nparams(sb) for sb in m.side_branches]
orders = len(branch_params)

for k in range(1, orders + 1):
    cum_layers = sum(layer_params[:k-1])
    branch_k = branch_params[k-1]

    eemlp_matched_cumulative_params.append(cum_layers + branch_k)

    inc_layer = layer_params[k-2] if k > 1 else 0
    eemlp_matched_params.append(inc_layer + branch_k)

pickle.dump({'x_vals': eemlp_matched_x_vals, 'y_vals_train': y_vals_eemlpmatched_train, 'y_vals_val': y_vals_eemlpmatched_val, 'y_vals_test': y_vals_eemlpmatched_test,
             'lr': lr, 'wd': wd, 'd_prob': d_prob, 'best_epoch': best_epoch, 'val_best': val_best,
             'params': eemlp_matched_params, 'cumulative_params': eemlp_matched_cumulative_params}, open(os.path.join(output_dir, 'eemlp-matched-vals.pkl'), 'wb'))