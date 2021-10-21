import cv2


def letter_box(image, size):
    h, w, _ = image.shape
    H, W = size
    scale = min(H / h, W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top = (H - new_h) // 2
    bottom = H - new_h - top
    left = (W - new_w) // 2
    right = W - new_w - left
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return new_image, scale, [top, bottom, left, right]