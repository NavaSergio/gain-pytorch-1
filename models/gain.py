import torch.nn as nn
import torch


class GAINGenerator(nn.Module):
    def __init__(self, dim, args, transform_params):
        super(GAINGenerator, self).__init__()
        self.args = args
        self.h_dim = dim
        norm_min = transform_params["min"]
        norm_max = transform_params["max"]
        self.register_buffer("norm_min", norm_min)
        self.register_buffer("norm_max", norm_max + 1e-6)
        self.model = None
        self.init_network(dim)
        self.seed_sampler = torch.distributions.Uniform(low=0, high=0.01)

    def init_network(self, dim):
        generator = [nn.Linear(dim * 2, dim), nn.ReLU()]
        generator.extend([nn.Linear(dim, dim), nn.ReLU()])
        generator.extend([nn.Linear(dim, dim), nn.Sigmoid()])
        for layer in generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.model = nn.Sequential(*generator)

    def normalizer(self, inp, mode="normalize"):
        if mode == "normalize":
            inp_norm = inp - self.norm_min
            inp_norm = inp_norm / self.norm_max

        elif mode == "renormalize":
            inp_norm = inp * self.norm_max
            inp_norm = inp_norm + self.norm_min

        else:
            raise NotImplementedError()

        return inp_norm

    def forward(self, data, mask):
        # MASK: gives you non-nans
        data_norm = self.normalizer(data)
        data_norm[mask == 0] = 0.
        z = self.seed_sampler.sample([data_norm.shape[0], self.h_dim]).to(self.args.device)
        random_combined = mask * data_norm + (1-mask) * z
        sample = self.model(torch.cat([random_combined, mask], dim=1))
        x_hat = random_combined * mask + sample * (1-mask)
        return sample, random_combined, x_hat


class GAINDiscriminator(nn.Module):
    def __init__(self, dim, label_dim, args):
        super(GAINDiscriminator, self).__init__()
        self.args = args
        self.hint_rate = torch.tensor(args.hint_rate)
        self.h_dim = dim
        self.discriminator = None
        self.init_network(dim)
        self.uniform = torch.distributions.Uniform(low=0, high=1.)

    def init_network(self, dim):
        discriminator = [nn.Linear(dim * 2, dim), nn.ReLU()]
        discriminator.extend([nn.Linear(dim, dim), nn.ReLU()])
        discriminator.extend([nn.Linear(dim, dim), nn.Sigmoid()])
        for layer in discriminator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.discriminator = nn.Sequential(*discriminator)

    def forward(self, x_hat, mask):
        hint = (self.uniform.sample([mask.shape[0], self.h_dim]) < self.hint_rate).float().to(self.args.device)
        hint = mask * hint
        inp = torch.cat([x_hat, hint], dim=1)
        return self.discriminator(inp)

