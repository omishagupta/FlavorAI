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
                    "text": "List only the ingredients visible in this image with the count in bullet points."
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


# Create Gradio interface
iface = gr.Interface(
    fn=extract_ingredients,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs="text",
    title="FlavorAI",
    description="Upload an image of food, and this tool will analyze it to extract the list of ingredients."
)

# Launch the Gradio app
iface.launch()
