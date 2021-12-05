import dataset.public_transforms as T
from dataset.voc import Voc
from dataset.coco import Coco
from torch.utils.data import DataLoader


class PublicTrainLoader:
    def __init__(self, cfg):
        self.dataset_name = cfg["Train"]["dataset_name"]
        self.batch_size = cfg["Train"]["batch_size"]
        self.voc_cfg = cfg["VOC"]
        self.coco_cfg = cfg["COCO"]
        self.input_size = cfg["Train"]["input_size"]
        self.max_num_boxes = cfg["Train"]["max_num_boxes"]
        self.transforms = [T.Resize(size=self.input_size),
                           T.TargetPadding(max_num_boxes=self.max_num_boxes),
                           T.ToTensor()]

    def _voc(self):
        return Voc(self.voc_cfg, T.Compose(transforms=self.transforms))

    def _coco(self):
        return Coco(self.coco_cfg, T.Compose(transforms=self.transforms))

    def __call__(self, *args, **kwargs):
        if self.dataset_name == "voc":
            return DataLoader(dataset=self._voc(), batch_size=self.batch_size, shuffle=True)
        elif self.dataset_name == "coco":
            return DataLoader(dataset=self._coco(), batch_size=self.batch_size, shuffle=True)
        else:
            raise ValueError("参数cfg->Train->dataset_name错误")
