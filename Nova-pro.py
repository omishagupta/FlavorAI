from PIL import Image
import gradio as gr
import boto3
import json
import base64
import re
import io
import numpy as np
import logging

# Initialize Boto3 client for Bedrock
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Change region if needed
boto3.set_stream_logger('botocore', level=logging.DEBUG)

def extract_ingredients(image):
    # Encode the image to base64
    pil_image = Image.fromarray(image)

    # Encode the image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")  # Save as JPEG in memory
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare the input prompt for the Bedrock model
    input_prompt = {
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpg",
                        "source": {"bytes": image_base64},
                    }
                },
                {
                    "text": "List only the ingredients visible in this image with the count in bullet points. Also provide heading identified ingredients in markdown heading"
                }
            ],
        }
    ],
        "inferenceConfig": {
            "max_new_tokens": 2048,
            "top_p": 0.9,
            "top_k": 20,
            "temperature": 0.7
        }
    }


    # Call the Bedrock model
    response = bedrock_client.invoke_model(
            modelId='amazon.nova-pro-v1:0',  # Updated model ID
            body=json.dumps(input_prompt),
            contentType='application/json',
            accept='application/json'
        )

    model_response = json.loads(response["body"].read())
    # Pretty print the response JSON.
    print(json.dumps(model_response, indent=2))
    # Print the text content for easy readability.
    ingredients = model_response["output"]["message"]["content"][0]["text"]
    print("\n[Response Content Text]")
    return ingredients

def generate_recipes(combined_input):
    try:
        input_prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"""Generate a recipe based on the following, please note that note that its not reuired to use all the ingredients:
                            Available ingredients: {combined_input}
                            
                            Please provide:
                            1. A simple recipe title
                            2. List of ingredients with measurements
                            3. Step-by-step cooking instructions
                            4. Estimated cooking time
                            5. Serving size"""
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": 4096,
                "top_p": 0.9,
                "top_k": 20,
                "temperature": 0.7
            }
        }

        response = bedrock_client.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps(input_prompt),
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        print(json.dumps(response_body, indent=2))
        # Print the text content for easy readability.
        recipe = response_body["output"]["message"]["content"][0]["text"]
        print("\n[Response Content Text]")
        return recipe
    
        if 'error' in response_body:
            return f"Error: {response_body['error']}"
            
        if 'output' in response_body:
            return response_body['output']
        
        return "No recipe generated"

    except Exception as e:
        return f"Error generating recipe: {str(e)}"

def process_input(image, additional_text):
    ingredients = extract_ingredients(image)
    # Combine detected ingredients with any additional ingredients/preferences
    combined_input = f"{ingredients}\nAdditional preferences: {additional_text}" if additional_text else ingredients
    recipe = generate_recipes(combined_input)
    return ingredients, recipe

def capture_image(img):
    try:
        if img is None:
            raise ValueError("No image captured!")
        return img, img
    except Exception as e:
        print(f"Capture error: {str(e)}")
        return None, None

def process_captured_image(captured_img, text_input):
    if captured_img is None:
        return "### Error\nNo image captured", "### Error\nPlease capture an image first"
    return process_input(captured_img, text_input)

with gr.Blocks(title="FlavorAI", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🍳 FlavorAI")
    gr.Markdown("Upload an image, capture from webcam, or provide ingredients to generate recipes.")
    
    # State variable to store captured image
    captured_image = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Tabs for different input methods
            with gr.Tabs():
                with gr.TabItem("Upload Image"):
                    upload_input = gr.Image(
                        type="numpy",
                        label="Upload an Image",
                        sources=["upload"],
                        height=350
                    )
                    
                    upload_analyze_btn = gr.Button("Analyze Uploaded Image", variant="primary")
                
                with gr.TabItem("Webcam"):
                    webcam_input = gr.Image(
                        type="numpy",
                        label="Capture from Webcam",
                        sources=["webcam"],
                        streaming=False,  # Changed to False
                        mirror_webcam=True,
                        height=350
                    )
                    
                    preview_image = gr.Image(
                        type="numpy",
                        label="Captured Image",
                        interactive=False,
                        height=200
                    )
                    
                    with gr.Row():
                        webcam_clear = gr.Button("Clear", variant="secondary", size="sm")
                        webcam_capture = gr.Button("Capture", variant="secondary", size="sm")
                    
                    webcam_analyze_btn = gr.Button("Analyze Captured Image", variant="primary")
            
            # Text input
            text_input = gr.Textbox(
                label="Additional Ingredients or Preferences",
                placeholder="Enter any additional ingredients or preferences (optional)",
                lines=2
            )
        
        with gr.Column(scale=1):
            ingredients_output = gr.Markdown(label="Detected Ingredients")
            recipe_output = gr.Markdown(label="Generated Recipe")
    
    # Webcam capture functionality
    def capture_image(img):
        if img is None:
            gr.Warning("No image captured!")
            return None, None
        return img, img

    webcam_capture.click(
        fn=capture_image,
        inputs=[webcam_input],
        outputs=[preview_image, captured_image]
    )
    
    # Clear webcam functionality
    webcam_clear.click(
        fn=lambda: (None, None, None),
        outputs=[webcam_input, preview_image, captured_image]
    )
    
    # Analysis functionality for uploaded images
    upload_analyze_btn.click(
        fn=process_input,
        inputs=[upload_input, text_input],
        outputs=[ingredients_output, recipe_output],
        api_name="analyze_upload",
        show_progress=True
    )
    
    # Analysis functionality for captured images
    webcam_analyze_btn.click(
        fn=process_captured_image,
        inputs=[captured_image, text_input],
        outputs=[ingredients_output, recipe_output],
        api_name="analyze_capture",
        show_progress=True
    )

if __name__ == "__main__":
    iface.launch()
