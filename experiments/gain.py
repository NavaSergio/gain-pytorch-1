import sys
sys.path.append("..")
import torch
import torch.optim as optim
from robustness.tools.helpers import AverageMeter
from tqdm import tqdm
import time
import os
import json
from models import GAINDiscriminator, GAINGenerator
import numpy as np


class GAINTrainer:

    def __init__(self, dim, label_dim, transform_params, args, load_path=None):
        self.dim = dim
        self.label_dim = label_dim
        self.discriminator = GAINDiscriminator(self.dim, self.label_dim, args)
        self.generator = GAINGenerator(self.dim, args, transform_params)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.alpha = args.alpha
        self.d_optimizer = optim.Adam(params=self.discriminator.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.Adam(params=self.generator.parameters(), lr=self.learning_rate)
        self.epoch = 0
        self.train_history = {"D_loss": [], "G_loss": [], "MSE_loss": []}
        self.eval_history = {}
        self.args = args
        self.result_dir = load_path if load_path else f"./results/GAIN/{args.data_type}/{time.strftime('%Y%m%d-%H%M%S')}"

    def discriminator_loss(self, mask, d_prob):
        d_loss = -torch.mean(mask * torch.log(d_prob+1e-8) + (1-mask) * torch.log(1-d_prob + 1e-8))
        return d_loss

    def generator_loss(self, mask, d_prob, random_combined, sample):
        g_loss = -torch.mean((1-mask) * torch.log(d_prob + 1e-8))
        mse_loss = torch.mean(torch.pow((mask * random_combined - mask*sample), 2)) / torch.mean(mask)
        return g_loss, mse_loss

    def save_checkpoint(self):
        ckpt_file = os.path.join(self.result_dir, "checkpoint")
        state_dict = dict()
        state_dict["generator"] = self.generator.state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["d_optimizer"] = self.d_optimizer.state_dict()
        state_dict["g_optimizer"] = self.g_optimizer.state_dict()
        state_dict["epoch"] = self.epoch
        torch.save(state_dict, ckpt_file)

    def train_step(self, loader):
        self.discriminator.train()
        self.generator.train()
        device = self.args.device
        loss_dict = dict()
        loss_dict["D_loss"] = AverageMeter()
        loss_dict["G_loss"] = AverageMeter()
        loss_dict["MSE_loss"] = AverageMeter()
        b_loader = tqdm(loader)
        for _, x_batch, _, m_batch in b_loader:
            x_batch, m_batch = x_batch.to(device), m_batch.to(device)
            self.g_optimizer.zero_grad()
            sample, random_combined, x_hat = self.generator(x_batch, m_batch)
            G_loss, mse_loss = self.generator_loss(m_batch, self.discriminator(x_hat, m_batch), random_combined, sample)
            generator_loss = G_loss + self.alpha * mse_loss
            generator_loss.backward()
            self.g_optimizer.step()

            self.d_optimizer.zero_grad()
            D_prob = self.discriminator(x_hat.detach(), m_batch)
            D_loss = self.discriminator_loss(m_batch, D_prob)
            D_loss.backward()
            self.d_optimizer.step()

            N = x_batch.shape[0]
            loss_dict["D_loss"].update(D_loss.detach().item(), N)
            loss_dict["G_loss"].update(G_loss.detach().item(), N)
            loss_dict["MSE_loss"].update(mse_loss.detach().item(), N)
            desc = []
            for k, v in loss_dict.items():
                desc.append(f"{k}: {v.avg:.4f}")
            b_loader.set_description(" ".join(desc))
        for k, v in loss_dict.items():
            loss_dict[k] = v.avg
        return loss_dict

    def rounding(self, tensor, masked_data):
        _, dim = tensor.shape
        rounded_data = tensor.clone()

        for i in range(dim):
            temp = masked_data[~torch.isnan(masked_data[:, i]), i]
            if len(torch.unique(temp)) < 20:
                rounded_data[:, i] = torch.round(rounded_data[:, i])
        return rounded_data

    def eval_model(self, loader, mode):
        device = self.args.device
        self.discriminator.eval()
        self.generator.eval()
        all_imputed = []
        all_orig = []
        all_mask = []
        all_data = []
        for x_original, x_batch, _, m_batch in loader:
            x_batch, m_batch, x_original = x_batch.to(device), m_batch.to(device), x_original.to(device)
            sample, random_combined, x_hat = self.generator(x_batch, m_batch)
            x_batch[m_batch == 0] = 0.
            imputed_data = m_batch * self.generator.normalizer(x_batch) + (1-m_batch) * sample
            all_imputed.append(imputed_data.detach())
            all_orig.append(x_original)
            all_mask.append(m_batch)
            all_data.append(x_batch)

        imputed_data = torch.cat(all_imputed, dim=0)
        x_original = torch.cat(all_orig, dim=0)
        mask = torch.cat(all_mask, dim=0)
        x = torch.cat(all_data, dim=0)
        imputed_data = self.generator.normalizer(imputed_data, mode="renormalize")
        x[mask == 0] = np.nan
        imputed_data = self.rounding(imputed_data, x)
        imputed_data = self.generator.normalizer(imputed_data)
        x_original = self.generator.normalizer(x_original)

        sq_error = torch.sum(((1 - mask) * x_original - (1 - mask) * imputed_data) ** 2)
        rmse = torch.sqrt(sq_error / ((1-mask).sum())).detach().cpu().item()
        return {mode+"-RMSE": rmse}

    def log_results(self, res_dict):
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        config = os.path.join(self.result_dir, "config.json")
        with open(config, 'w') as fp:
            json.dump(vars(self.args), fp)

        perf_file = os.path.join(self.result_dir, "performance.json")
        with open(perf_file, "w") as fp:
            json.dump(res_dict, fp)

    def train_model(self, train_loader, test_loader):
        device = self.args.device
        self.discriminator.to(device)
        self.generator.to(device)

        t_loader = tqdm(range(self.args.max_epochs))

        for i in t_loader:
            summary = self.train_step(train_loader)
            for key, val in summary.items():
                self.train_history[key].append(val)
            self.epoch += 1

            desc = list([f"Epoch: {self.epoch}"])
            for k, v in summary.items():
                desc.append(f"{k}: {v:.3f}")
            desc = " ".join(desc)
            t_loader.set_description(desc)

            if (self.epoch + 1) % self.args.eval_freq == 0:
                train_eval = self.eval_model(train_loader, mode="train")
                test_eval = self.eval_model(test_loader, mode="test")
                print(json.dumps(train_eval))
                print(json.dumps(test_eval))

        train_eval = self.eval_model(train_loader, mode="train")
        test_eval = self.eval_model(test_loader, mode="test")
        perf_dict = train_eval
        perf_dict.update(test_eval)
        self.log_results(perf_dict)
        self.save_checkpoint()
        return perf_dict
