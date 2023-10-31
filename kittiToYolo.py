import fiftyone as fo
import fiftyone.utils.kitti

# %%
dataset_dir = "yolov8-test/datasets/"
fiftyone.utils.kitti.download_kitti_detection_dataset( dataset_dir , overwrite=True , cleanup=True )

# %%
print(fo.list_datasets())
# %%
# DELETE DATA
# dataset = fo.load_dataset("my-dataset-kitti0723")#my-dataset-kitti0709
# dataset.delete()
# %%
# import kitti dataset
name = "my-dataset-kitti0723_1222"
dataset_dir = "yolov8-test/datasets/data_tracking_image_2/training"
# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.KITTIDetectionDataset,
    name=name,
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())
print("done")
# %%
print(fo.list_datasets())

# %%
# convert

output_yolo_dir = "/home/yz/PycharmProjects/2022/TRUST-E/ErrorResilienceAnalysisYOLOV3/0723ConvertToYOLO/"
# dataset_type=fo.types.YOLOv4Dataset
label_field = "ground_truth"
dataset.export(
    export_dir=output_yolo_dir,
    dataset_type=fo.types.YOLOv4Dataset,
    label_field=label_field,
)
# fiftyone.utils.data.converters.convert_dataset( input_dir=dataset_dir, input_type=fo.types.KITTIDetectionDataset, input_kwargs=None , dataset_importer=None , output_dir=output_yolo_dir, output_type=fo.types.YOLOv4Dataset , output_kwargs=None , dataset_exporter=None , overwrite=False )
