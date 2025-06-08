from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model

# Predict with the model
results = model("/local_datasets/VLM_input/c013f92386ab5921.jpg")  # predict on an image