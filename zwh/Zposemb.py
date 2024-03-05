import torch
import matplotlib.pyplot as plt


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(len, dim, temperature: int = 10000, dtype=torch.float32):
    x = torch.arange(len)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)

    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos()), dim=1)  # 这里不用担心,不交叉无所谓,
    return pe.type(dtype)


if __name__ == '__main__':

    pos = posemb_sincos_1d(200, 256)
    # pos = posemb_sincos_2d(20,20,256)

    # 创建一个热力图
    plt.imshow(pos, cmap='hot', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    # 显示图形
    plt.show()
    pass
