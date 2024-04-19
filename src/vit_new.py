# Code from the awesome lucidrains: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
# with some of my own modifications

import torch
from torch import nn
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from einops import rearrange
from src.he2rna import compute_correlations
import pdb
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, num_outputs, dim, depth, heads, mlp_dim, dim_head = 64,
                 num_clusters=100, device='cuda'):
        super().__init__()

        self.pos_emb1D = nn.Parameter(torch.randn(num_clusters, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_outputs)
        )
        self.device = device

    def forward(self, x):
        #pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + self.pos_emb1D

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

def train(model, dataloaders, optimizer, accelerator=None,
          num_epochs=200, save_dir='exp/', patience=20,
          run=None, verbose=True, phases=['train', 'val'], split=None,
          save_on='loss', stop_on='loss', delta=0.5):

    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if split:
        save_path = os.path.join(save_dir, f'model_best_{split}.pt')
    else:
        save_path = os.path.join(save_dir, 'model_best.pt')

    loss_fn = nn.MSELoss()
    epoch_since_best = 0
    best_loss = np.inf

    # these are for early stopping on loss + score
    early_stop_on_loss_triggered = 0
    epoch_since_best_score = 0
    best_score = 0
    epoch_since_ok_loss = 0

    for epoch in tqdm(range(num_epochs)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            losses = {
                'train': [],
                'val': []
            }
            maes = {
                'train': [],
                'val': []
            }
            scores = {
                'train': [],
                'val': []
            }

            for s, (image, rna_data, _, _) in enumerate(dataloaders[phase]):
                if image == []: continue
                image = image.to(model.device)
                rna_data = rna_data.to(model.device)

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(image)

                loss = loss_fn(pred, rna_data)
                mae = mean_absolute_error(rna_data.detach().cpu().numpy(), pred.detach().cpu().numpy())
                score = compute_correlations(rna_data.detach().cpu().numpy(), pred.detach().cpu().numpy())

                losses[phase] += [loss.detach().cpu().numpy()]
                maes[phase] += [mae]
                scores[phase] += [score]

                if phase == 'train':
                    optimizer.zero_grad()
                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    optimizer.step()

            losses[phase] = np.mean(losses[phase])
            maes[phase] = np.mean(maes[phase])
            scores[phase] = np.mean(scores[phase])

            if phase == 'val':
                suffix = 'id'
            else:
                suffix = ''

            if run:
                run.log({'epoch': epoch, f'score {phase}{suffix} {split}': scores[phase]})
                run.log({'epoch': epoch, f'{phase}{suffix} loss fold {split}': losses[phase]})
                run.log({'epoch': epoch, f'{phase}{suffix} mae fold {split}': maes[phase]})

            if verbose:
                print(f'Epoch {epoch}: {phase} loss {losses[phase]} mae {maes[phase]}')

            if (phase == 'val') or (len(phases) == 1):

                # only relevant for early stopping on loss+corr
                if early_stop_on_loss_triggered == 1:
                    if losses[phase] < (best_loss + delta): # we allow loss to deviate a little bit from optimal while continuing training for good correlation
                        epoch_since_ok_loss = 0
                    else:
                        epoch_since_ok_loss += 1

                # relevant for both early stopping and model save on loss/loss+corr
                if losses[phase] < best_loss:
                    best_loss = losses[phase]
                    epoch_since_best = 0
                    if save_on == 'loss':
                        torch.save(model.state_dict(), save_path)
                    elif (save_on == 'loss+corr') and (early_stop_on_loss_triggered == 0): # first save model based on loss, later overwrite if there is a model at epoch with loss close to best loss and better correlation
                        torch.save(model.state_dict(), save_path)
                else:
                    epoch_since_best += 1

                # for early stopping and model save based on loss+corr
                if scores[phase] > best_score:
                    best_score = scores[phase]
                    epoch_since_best_score = 0
                    if (save_on == 'loss+corr') and (early_stop_on_loss_triggered == 1):
                        torch.save(model.state_dict(), save_path)
                        print(f'Saved model on loss+corr at epoch {epoch} of better score and loss within {delta} of optimal loss')
                else:
                    epoch_since_best_score += 1

        if epoch_since_best == patience:
            early_stop_on_loss_triggered = 1
            if stop_on == 'loss':
                print(f'Early stopping at epoch {epoch}!')
                break

        if stop_on == 'loss+corr':
            if (early_stop_on_loss_triggered == 1) and (epoch_since_best_score == patience):
                print(f'Early stopping at epoch {epoch} because neither loss nor score is improving anymore!')
                break

            if (early_stop_on_loss_triggered == 1) and (epoch_since_ok_loss == patience):
                print(f'Early stopping at epoch {epoch} because loss is not within {delta} of best loss anymore!')
                break
    return model

def evaluate(model, dataloader, run=None, verbose=True, suff=''):
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    preds = []
    real = []
    wsis = []
    projs = []
    maes = []
    smapes = []
    for image, rna_data, wsi_file_name, tcga_project in tqdm(dataloader):

        if image == []: continue

        image = image.to(model.device)
        rna_data = rna_data.to(model.device)
        wsis.append(wsi_file_name)
        projs.append(tcga_project)

        pred = model(image)
        preds.append(pred.detach().cpu().numpy())
        loss = loss_fn(pred, rna_data)
        real.append(rna_data.detach().cpu().numpy())
        mae = mean_absolute_error(rna_data.detach().cpu().numpy(), pred.detach().cpu().numpy())
        smape_var = smape(rna_data.detach().cpu().numpy(), pred.detach().cpu().numpy())
        losses += [loss.detach().cpu().numpy()]
        maes += [mae]
        smapes += [smape_var]

    losses = np.mean(losses)
    maes = np.mean(maes)
    smapes = np.mean(smapes)
    if run:
        run.log({f'test_loss'+suff: losses})
        run.log({f'test_MAE'+suff: maes})
        run.log({f'test_MAPE'+suff: smapes})
    if verbose:
        print(f'Test loss: {losses}')
        print(f'Test MAE: {mae}')
        print(f'Test MAPE: {smapes}')

    preds = np.concatenate((preds), axis=0)
    real = np.concatenate((real), axis=0)
    wsis = np.concatenate((wsis), axis=0)
    projs = np.concatenate((projs), axis=0)

    return preds, real, wsis, projs

def predict(model, dataloader, run=None, verbose=True):
    model.eval()
    preds = []
    wsis = []
    projs = []
    for image, rna_data, wsi_file_name, tcga_project in tqdm(dataloader):
        if image == []: continue
        image = image.to(model.device)
        wsis.append(wsi_file_name)
        projs.append(tcga_project)

        pred = model(image)
        preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate((preds), axis=0)
    wsis = np.concatenate((wsis), axis=0)
    projs = np.concatenate((projs), axis=0)

    return preds, wsis, projs


##### functions for pathway prediction
def train_pathway(model, dataloaders, optimizer, accelerator=None,
                    num_epochs=200, save_dir='exp/', patience=20,
                    run=None, verbose=True, phases=['train', 'val'], split=None, model_type='vit'):

    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if split:
        save_path = os.path.join(save_dir, f'model_best_{split}.pt')
    else:
        save_path = os.path.join(save_dir, 'model_best.pt')

    loss_fn = nn.BCEWithLogitsLoss()
    epoch_since_best = 0
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            losses = { 'train': [], 'val': [] }

            for s, (image, label, _, _) in enumerate(dataloaders[phase]):
                if image == []: continue
                image = image.to(model.device)
                label = label.to(model.device).unsqueeze(1).float()

                if model_type == 'he2rna':
                    image = rearrange(image, 'b c f -> b f c')
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(image)

                loss = loss_fn(pred, label)
                losses[phase] += [loss.detach().cpu().numpy()]

                if phase == 'train':
                    optimizer.zero_grad()
                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    optimizer.step()

            losses[phase] = np.mean(losses[phase])

            if run:
                run.log({'epoch': epoch, f'{phase} loss fold {split}': losses[phase]})

            if verbose:
                print(f'Epoch {epoch}: {phase} loss {losses[phase]}')

            if (phase == 'val') or (len(phases) == 1):

                if losses[phase] < best_loss:
                    best_loss = losses[phase]
                    epoch_since_best = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    epoch_since_best += 1

        if epoch_since_best == patience:
            print(f'Early stopping at epoch {epoch}!')
            break

    return model


def evaluate_pathway(model, dataloader, run=None, verbose=True, suff='', model_type='vit'):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    preds = []
    real = []
    wsis = []
    projs = []

    for image, label, wsi_file_name, tcga_project in tqdm(dataloader):

        if image == []: continue

        image = image.to(model.device)
        label = label.to(model.device).unsqueeze(1).float()
        wsis.append(wsi_file_name)
        projs.append(tcga_project)

        if model_type == 'he2rna':
            image = rearrange(image, 'b c f -> b f c')

        out = model(image)
        loss = loss_fn(out, label)
        losses += [loss.detach().cpu().numpy()]

        pred = torch.sigmoid(out) ### important to add sigmoid here
        preds.append(pred.detach().cpu().numpy())
        real.append(label.detach().cpu().numpy())

    losses = np.mean(losses)
    if run:
        run.log({f'test_loss'+suff: losses})
    if verbose:
        print(f'Test loss: {losses}')

    preds = np.concatenate((preds), axis=0)
    real = np.concatenate((real), axis=0)
    wsis = np.concatenate((wsis), axis=0)
    projs = np.concatenate((projs), axis=0)

    return preds, real, wsis, projs
