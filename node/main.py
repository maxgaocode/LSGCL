#coding=utf-8
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from LSGCL import LSGCL
from eval import label_classification

import numpy as np

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
'''
def train(model, data, Norm):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(data.x, drop_feature_rate_1)
    x_2 = drop_feature(data.x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return z1, loss.item()

'''
def train(model, data, Norm):
    model.train()
    optimizer.zero_grad()
    mo = t()
    z1 = model(data.x, data.edge_index, Norm)
    
    loss= model.loss(z1, batch_size=0)
    loss.backward()
    optimizer.step()

    return z1, loss.item()


def test(model, data, Norm, final=False):
    model.eval()
    x, edge_index, y=data.x, data.edge_index, data.y,
    z = model(x, edge_index, Norm)

    micro, macro = label_classification(z, y, ratio=0.1)
    return micro, macro
#################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Photo')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--Norm', type=int, default=1)
    parser.add_argument('--net', type=str, default='LSGCL')#LightGCL
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    if args.K>1:
        
        torch.manual_seed(12345)
        random.seed(12345)
        num_layers = args.K-1
        num_epochs = args.num_epochs
        eachdim = 64

    from datasets import *

    if args.dataset in ['pubmed']:
        dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
        data = dataset[0]

    elif args.dataset in ['Computers', 'Photo']:
        dataset = get_amazon_dataset(args.dataset, args.normalize_features)
        data = dataset[0]  # Data(edge_index=[2, 163788], x=[18333, 6805], y=[18333])
    elif args.dataset == 'CS' or args.dataset == 'Physics':
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
        data = dataset[0]  # Data(edge_index=[2, 163788], x=[18333, 6805], y=[18333])


    elif args.dataset in ['film']:
        dataset = dataset_heterophily(root='../datacon/', name=args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]


    elif args.dataset in ['chameleon', 'squirrel', 'texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../datacon/', name=args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data = data.to(device)


    print(args.dataset)
    print(data)
   

    model = LSGCL(data.x.shape[1], eachdim, args.hidden, args.tau, num_layers, activation="relu").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        z1, loss = train(model, data, args.Norm)
        now = t()
        if epoch % 5 == 0:
            
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')

        prev = now
    totaltime = t() - start
    acc = []
    for i in range(10):
        micro, macro = test(model, data, args.Norm, final=True)
        print(micro, macro)
        acc.append(micro)
    print(acc)
    meanmicro = sum(acc) / 10
    m1 = np.std(acc)


    print("=== Final ===")
    print(args.dataset)
    filename = f'{args.net}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write("net:{}, dataset:{}, lr:{}, weight_decay:{}, Norm:{},  K:{}, tau:{}, epoch:{}, meanmicro:{}, std:{} "
                        .format(args.net, args.dataset, args.lr, args.weight_decay, args.Norm, args.K, args.tau, args.num_epochs,
                                meanmicro, m1,  totaltime))
        write_obj.write("\n")

