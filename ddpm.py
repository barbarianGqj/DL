# define the class DDPM, which defines the process of forward and backward
import torch


class DDPM():
    def __init__(self,
                 device,
                 n_steps,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1-betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.n_steps = n_steps

    def sample_forward(self, x, t, eps=None):
        """
        forward progress
        add noise
        t.shape => (batch_size, ) but alpha_bars.shape => 一维tensor
        """
        alpha_bar = self.alpha_bars[t].reshape(-1,1,1,1)  # (batch_size, 1, 1, 1) because img is 4-D
        if eps is None:   # normal distribution
            eps = torch.randn_like(x)
        res = torch.sqrt(1- alpha_bar) * eps + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_grad=True):
        """
        one step of backward pass
        """
        n = x_t.shape[0]
        t_tensor = torch.tensor([t]*n, dtype = torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1- self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t

    def sample_backward(self, img_shape, net, device, simple_var=True, clip_grad=True):
        x = torch.rand(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var=simple_var, clip_grad=clip_grad)
        return x
