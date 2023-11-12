# import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from bdhnet.data.datasets.register_bdh_instance import register_all_bdh_instance
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from evaluation.bdh_coco_evaluation import BDHCOCOEvaluator

from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
from detectron2.projects.deeplab import add_deeplab_config
from bdhnet import add_maskformer2_config, BDHInstanceNewBaselineDatasetMapper

args = default_argument_parser().parse_args()
# load the config file, configure the threshold value, load weights
cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.freeze()
default_setup(cfg, args)


# Create predictor
predictor = DefaultPredictor(cfg)

# Call the COCO Evaluator function and pass the Validation Dataset
evaluator = BDHCOCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR, max_dets_per_image=200)

val_mapper = BDHInstanceNewBaselineDatasetMapper(cfg, False)
val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=val_mapper)


# Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)
print(0)
