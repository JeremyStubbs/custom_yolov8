As I did with mobilenet and resnet, I customize a well known image classification model. 

boundary_boxes_custom.py uses a pretrained yolov8 to make bounding boxes arouond the identified object.

classification_custom.py will build the yolov8 model from scratch so that we can gain full functionality locally without using their API.
