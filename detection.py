from imageai.Detection import ObjectDetection
import os

# Get the current working directory
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "test-images/10.jpg"),
    output_image_path = os.path.join(execution_path, "test-images/10-detection.jpg"),
    minimum_percentage_probability=50, # level of confidence, will only detect things that has a 50% confidence rating
)

for eachObject in detections:
    print(
        eachObject["name"],
        " : ",
        eachObject["percentage_probability"]
    )
