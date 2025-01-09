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

# Create Gradio interface with both outputs
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Textbox(
        label="Additional Ingredients or Preferences",
        placeholder="Enter any additional ingredients or preferences (optional)",
        lines=2
        )
    ],
    outputs=[
        gr.Textbox(label="Detected Ingredients"),
        gr.Textbox(label="Generated Recipe")
    ],
    title="FlavorAI",
    description="Upload an image of food to detect ingredients and generate recipes."
)

# Launch the Gradio app
iface.launch()
