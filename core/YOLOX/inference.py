import torch
import torchvision

from utils.tools import reverse_direct_image_resize


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        print("class_conf范围：", torch.min(class_conf), torch.max(class_conf))
        print("image_pred[:, 4]范围： ", torch.min(image_pred[:, 4]), torch.max(image_pred[:, 4]))

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        print(detections, detections.size())
        print("conf_thres: ", conf_thre)
        print("conf:", conf_mask)
        print(torch.any(conf_mask))
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections), dim=0)

    return output


def get_specific_detection_results(dets, H, W, input_image_size):
    """

    Args:
        dets: torch.Tensor, shape: [N, 6(xmin, ymin, xmax, ymax, obj_conf, class_conf, class_pred)]
        H: 待检测的图片原始高度
        W: 待检测的图片原始宽度
        input_image_size: Int

    Returns:

    """
    if dets is None:
        return None, None, None
    boxes = reverse_direct_image_resize(h=H, w=W, input_size=input_image_size, boxes=dets[:, :4], xywh=False,
                                        coords_normalized=True)
    scores = dets[:, 4] * dets[:, 5]
    clses = dets[:, 6].to(torch.int32)
    return boxes, scores, clses
