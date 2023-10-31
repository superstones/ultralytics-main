import argparse
import os

from ultralytics import YOLO

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
    api_key="lOWRj59X5bEjrsc1UCsIOGXUQ",
    project_name="ultralytics-main",
    workspace="292454993-qq-com"
)

# Load a model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model = YOLO('yolov8x.yaml')  # build a new model from YAML

    model = YOLO('ultralytics/cfg/models/v8/my_yolov8.yaml')  # load a pretrained model (recommended for training)

    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8x.pt')  # build from YAML and transfer weights
    results = model.train(data='Argoverse.yaml', epochs=100, imgsz=640, cfg='default_copy.yaml')
    hyper_params = {
        "learning_rate": 0.01,
        "steps": 100000,
        "batch_size": 50,
    }
    experiment.log_parameters(hyper_params)

    # Initialize and train your model
    # model = TheModelClass()
    # train(model)

    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="yolov8n")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/crack.yaml', help='dataset.yaml path')
    # parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    # parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs, -1 for autobatch')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--rect', action='store_true', help='rectangular training')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Lion'], default='Lion',
    #                     help='optimizer')
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
# 0)
