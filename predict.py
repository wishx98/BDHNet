import sys

sys.path.insert(0, "BDHNET")
import tempfile
from pathlib import Path
import numpy as np
import cv2
import cog

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from bdhnet import add_maskformer2_config


class Predictor(cog.Predictor):
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("/home/swx/local/codes/lr_dt2/configs/BDH/instance-segmentation/swin/BDHNet_swin_small_bs16_50ep.yaml")
        cfg.MODEL.WEIGHTS = "/home/swx/local/codes/BDHNet/pretrained_weights/phase2.pth"
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("bdh_val")

    @cog.input(
        "image",
        type=Path,
        help="Input image for segmentation. Output will be the concatenation of Panoptic segmentation (top), "
        "instance segmentation (middle), and semantic segmentation (bottom).",
    )
    def predict(self, image):
        im = cv2.imread(str(image))
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), result)
        return out_path
