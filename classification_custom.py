from ultralytics import YOLO
from ultralytics import settings

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to TF SavedModel format
model.export(format="saved_model")  

# Load the exported TF SavedModel model
tf_savedmodel_model = YOLO("./yolo11n.pt")

import onnx
import onnx_tf

# Load ONNX model
onnx_model = onnx.load('yolo11n.onnx')

# Convert ONNX model to TensorFlow format
tf_model = onnx_tf.backend.prepare(onnx_model)

# Export TensorFlow model
tf_model.export_graph("yolo11n.pb") 


# Run inference
results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")
