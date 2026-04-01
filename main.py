import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
from network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512
n_epochs = 100

def train(ddpm: DDPM, net, device, ckpt_path):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss(reduction='sum')  # 使用MSE时，“会导致损失值过小，部分参数的梯度可能会被忽略为0，从而导致训练过程先收敛后发散”
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


    for epoch in range(n_epochs):
        tic = time.time()
        total_loss = 0
        for x,_ in dataloader:
            current_batch_size = x.shape[0]
            # sample
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)  # sample from [0, n_steps - 1]
            eps = torch.randn_like(x).to(device)
            # forward and predict
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            # optimize
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {epoch} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print("Done")


# generate imgs
def sample_imgs(ddpm,
                net,
                output_path,
                n_sample = 81,
                device = 'cuda',
                simple_var=True
                ):
    net = net.to(device)  
    net = net.eval()  # test mode
    with torch.no_grad():  # close calculation of gradient
        shape = (n_sample, *get_img_shape())   # the shape of the imgs to be generated
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()  # (81, 1, 28, 28)
        imgs = (imgs + 1) / 2 * 255    # from (-1, 1) to (0, 255)
        imgs = imgs.clamp(0, 255)  # ensure value is in (0, 255)
        imgs = einops.rearrange(imgs,   # 小图拼接成大图
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))
        imgs = imgs.numpy().astype(np.uint8)  # 将张量转为opencv支持的格式
        cv2.imwrite(output_path, imgs)

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = 'model_path/model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)
    # for training
    # train(ddpm, net, device=device, ckpt_path=model_path)

    # for testing
    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)



