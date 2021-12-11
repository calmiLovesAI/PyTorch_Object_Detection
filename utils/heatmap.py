import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from utils.tools import cv2_read_image


def visualize_heatmap(image_path, heatmap, output_dir=None, channel_format="first"):
    """
    可视化heatmap
    :param image_path: 原始图片路径
    :param heatmap: torch.Tensor, (N, C, H, W) or (N, H, W, C) 热图
    :param output_dir: 可视化图片的保存路径
    :param channel_format: str, 取值"first"或"last", 根据heatmap的channel在哪一个维度来
    :return:
    """
    plt.jet()
    image, h, w, c = cv2_read_image(image_path)
    plt.subplots(nrows=1, ncols=1)
    plt.imshow(image, alpha=1)

    if channel_format == "last":
        heatmap = torch.permute(heatmap, dims=(0, 2, 3, 1))  # (N, H, W, C) -> (N, C, H, W)
    N, C, H, W = heatmap.size()
    heatmap = torch.reshape(heatmap, shape=(-1, H, W))
    heatmap = torch.sum(heatmap, dim=0, keepdim=True)
    heatmap = torch.sigmoid(heatmap)

    resized_heatmap = TF.resize(img=heatmap, size=[h, w])
    resized_heatmap = resized_heatmap.squeeze().numpy()
    plt.imshow(resized_heatmap, alpha=0.5)
    if output_dir is not None:
        plt.savefig(output_dir)
    plt.show()


