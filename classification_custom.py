from ultralytics import YOLO
import time

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# print('here')
# time.sleep(10)

# Train the model
results = model.train(data='custom_data', epochs=100, imgsz=224)



"""
In root directory there is a folder called datasets. It contains a folder named custom_data which contains a train folder and a test folder. 
Each contain folders for each class which contain the images. 


"""