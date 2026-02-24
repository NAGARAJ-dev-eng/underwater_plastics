# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import base64

# app = Flask(__name__)

# # Load the YOLOv8 model
# model = YOLO(r'C:\Users\nagar\OneDrive\Desktop\plastic detection\underwater_plastics\runs\detect\custom_yolov8s_model_50\weights\best (2).pt')  # Update the path if necessary

# @app.route('/')
# def index():
#     """Render the home page."""
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     """Handle file upload and run object detection."""
#     file = request.files.get('image')  # Get the uploaded file
#     if not file:
#         return "No file uploaded", 400

#     # Convert uploaded image to a numpy array
#     file_bytes = np.frombuffer(file.read(), np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Run YOLOv8 inference
#     results = model.predict(source=image, imgsz=640, conf=0.5, save=False)

#     # Retrieve predictions
#     detections = []
#     for box in results[0].boxes:
#         class_id = int(box.cls)
#         confidence = float(box.conf)
#         detections.append((model.names[class_id], round(confidence * 100, 2)))

#     # Annotate the image
#     annotated_image = results[0].plot()

#     # Convert annotated image to a base64 string
#     _, buffer = cv2.imencode('.jpg', annotated_image)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')

#     # Render the result page with the image and detection results
#     return render_template('result.html', image_data=img_base64, detections=detections)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLOv8 models
model_1 = YOLO(r'D:\\Project\\underwater_plastics\\runs\\detect\\custom_yolov8s_model_50\\weights\\best (2).pt')  # First model
model_2 = YOLO(r'D:\\Project\\underwater_plastics\\runs\\detect\\custom_yolov8l_model_100\\weights\\best (1).pt') #Second model

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle file upload and run object detection with both models."""
    file = request.files.get('image')  # Get the uploaded file
    if not file:
        return "No file uploaded", 400

    # Convert uploaded image to a numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run inference with the first model
    results_1 = model_1.predict(source=image, imgsz=640, conf=0.5, save=False)
    detections_1 = []
    for box in results_1[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        detections_1.append((model_1.names[class_id], round(confidence * 100, 2)))

    # Annotate the image for the first model
    annotated_image_1 = results_1[0].plot()

    # Run inference with the second model
    results_2 = model_2.predict(source=image, imgsz=640, conf=0.5, save=False)
    detections_2 = []
    for box in results_2[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        detections_2.append((model_2.names[class_id], round(confidence * 100, 2)))

    # Annotate the image for the second model
    annotated_image_2 = results_2[0].plot()

    # Convert both annotated images to base64 strings
    _, buffer_1 = cv2.imencode('.jpg', annotated_image_1)
    img_base64_1 = base64.b64encode(buffer_1).decode('utf-8')

    _, buffer_2 = cv2.imencode('.jpg', annotated_image_2)
    img_base64_2 = base64.b64encode(buffer_2).decode('utf-8')

    # Render the result page with both model results
    return render_template(
        'result.html', 
        image_data_1=img_base64_1, 
        image_data_2=img_base64_2, 
        detections_1=detections_1, 
        detections_2=detections_2
    )

if __name__ == '__main__':
    app.run(debug=True)

