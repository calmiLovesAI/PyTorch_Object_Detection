import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from utils.tools import cv2_read_image


def visualize_heatmap(image_path, heatmap, output_dir=None):
    """
    可视化heatmap
    :param image_path: 原始图片路径
    :param heatmap: torch.Tensor, (1, H, W)  热图
    :param output_dir: 可视化图片的保存路径
    :return:
    """
    plt.jet()
    image, h, w, c = cv2_read_image(image_path)
    plt.subplots(nrows=1, ncols=1)
    plt.imshow(image, alpha=1)

    resized_heatmap = TF.resize(img=heatmap, size=[h, w])
    resized_heatmap = resized_heatmap.squeeze().numpy()
    plt.imshow(resized_heatmap, alpha=0.5)
    if output_dir is not None:
        plt.savefig(output_dir)
    plt.show()


