from ultralytics import YOLO
import cv2


im1 = cv2.imread("im1.jpg")

model = YOLO('yolov8n.pt')

# print(type(model))

results = model.predict("im1.jpg", save=False)  


print(results[0].names[5])
# print('here', results[0].boxes)


# default `log_dir` is "runs" 
# writer = SummaryWriter()
# writer.add_graph(model, im1)
# writer.close()

 
for result in results:
    # Detection
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)

    # Segmentation
    # result.masks.data      # masks, (N, H, W)
    # result.masks.xy        # x,y segments (pixels), List[segment] * N
    # result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # Classification
    # result.probs     # cls prob, (num_class, )

"""
To train a model that does bounding boxes, the folder structure is train which 
has images/train and labels/train folders. The labels/train folder contains text files with same 
names as images in images/train folder. Each line of the text file contains object 
class and bounding box (unknown coordinates x1, y1, x2, y2?).
# ___________________________________
According to the training commands, we will execute further in this article, 
this YAML file should be in the project root directory. We will name this file pothole_v8.yaml.

# yamnl contents _________________________________
path: pothole_dataset_v8/
train: 'train/images'
val: 'valid/images'
 
# class names
names: 
  0: 'pothole'

# Train with one of two options below______________________________
results = model.train(data='custom_data.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   name='yolov8n_custom')

# Can also do on CLI
# yolo task=detect mode=train model=yolov8n.pt imgsz=640 data=custom_data.yaml epochs=10 batch=8 name=yolov8n_custom

"""