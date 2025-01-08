import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image

# Load the saved model and tokenizer
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")

# Get the list of labels from the model's configuration
labels = list(model.config.id2label.values())

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "/Users/omishagp/Workspace/FlavorAI/1658311473_mixvegetable.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image)  # Convert NumPy array to PIL image
input_tensor = preprocess(pil_image).unsqueeze(0)

# Run the image through the model
outputs = model(input_tensor)

# Get the predicted label index
predicted_idx = torch.argmax(outputs.logits, dim=1).item()

# Get the predicted label text
predicted_label = labels[predicted_idx]

# Print the predicted label
print("Detected label:", predicted_label)
