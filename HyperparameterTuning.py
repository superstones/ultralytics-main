from ultralytics import YOLO
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
    api_key="lOWRj59X5bEjrsc1UCsIOGXUQ",
    project_name="ultralytics-main",
    workspace="292454993-qq-com"
)

if __name__ == '__main__':
    # Initialize the YOLO model
    model = YOLO('ultralytics/cfg/models/v8/my_yolov8.yaml')

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(data='Argoverse.yaml', epochs=60, iterations=300, optimizer='AdamW', plots=True, save=True, val=False)
    # Report multiple hyperparameters using a dictionary:
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
