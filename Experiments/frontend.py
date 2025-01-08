import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("YOLOv8m-trained-Model/weights/best.pt")

# Define the prediction function
def predict(image):
    try:
        # Convert uploaded image to numpy array
        img_array = np.array(image)

        # Run YOLO inference
        results = model.predict(source=img_array, conf=0.1)

        # Annotate image with results
        annotated_img = results[0].plot()  # Get the annotated image

        # Extract detection results with error checking
        detections = []
        for box in results[0].boxes:
            try:
                cls_value = box.cls
                if isinstance(cls_value, (list, np.ndarray)):
                    cls_value = cls_value[0]  # Take first element if it's a list/array
                
                conf_value = box.conf
                if isinstance(conf_value, (list, np.ndarray)):
                    conf_value = conf_value[0]

                detection = {
                    "class": int(cls_value),
                    "confidence": round(float(conf_value), 2),
                    "bbox": [int(x) for x in box.xyxy[0].tolist()]
                }
                detections.append(detection)
            except Exception as e:
                print(f"Error processing box: {e}")
                continue

        return annotated_img, detections

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, []


# Create Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.CheckboxGroup(
            choices=["Eggs", "Wheat Flour"],
            label="Select Options",
            info="Choose one or more options"
        ),
        gr.Textbox(label="Additional Notes", placeholder="Enter any additional information")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected Image"),    # Show annotated image
        gr.JSON(label="Detection Details")                # Display detection details
    ],
    # live=True,  # Enable live processing for webcam/video
    title="FlavorAI",
    description="test."
)

# Launch the app
iface.launch()
