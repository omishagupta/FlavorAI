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
# boto3.set_stream_logger('botocore', level=logging.DEBUG)

def extract_ingredients(media):

    try:
        # Check if media is a video file path (string)
        if isinstance(media, str):
            with open(media, "rb") as video_file:
                    binary_data = video_file.read()
                    base_64_encoded_data = base64.b64encode(binary_data)
                    base64_string = base_64_encoded_data.decode("utf-8")
                # Define your system prompt(s).
            system_list = [
                {
                    "text": f"""You are an expert media analyst and a professional in identifying food ingredients from visual content."""
                }
            ]
                # Define a "user" message including both the image and a text prompt.
            input_prompt = {
                "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": "mp4",
                                "source": {"bytes": base64_string},
                            }
                        },
                        {
                            "text": f"""You are an expert media analyst and a professional in identifying food ingredients from visual content. When analyzing the provided video, adhere to the following instructions:
                                        List only the exact ingredients visible in the video. Be specific in naming the ingredients (e.g., "red pepper" instead of "bell pepper"). Avoid vague terms or generalizations.
                                        Exclude non-edible items. Focus only on the edible ingredients.
                                        Provide the ingredients and their count as bullet points. Each point should include:
                                        The name of the ingredient.
                                        The quantity, if it is visually identifiable.
                                        Present your findings under the heading "Identified Ingredients" using markdown format (e.g., ## Identified Ingredients).
                                        Do not include any commentary or observations outside of the requested format. Only list edible ingredients that can be clearly identified in the video."""
                        },
                    ],
                }
            ] }
            # Configure the inference parameters.
            inf_params = {"max_new_tokens": 300, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

            native_request = {
                "schemaVersion": "messages-v1",
                "messages": input_prompt,
                "system": system_list,
                "inferenceConfig": inf_params,
            }
            
        # Handle image input (numpy array)
        else:
            print("Processing image...")
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(media)

            # Encode the image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            input_prompt = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpg",
                                    "source": {"bytes": image_base64}
                                }
                            },
                            {
                                "text": "List only the ingredients visible in this image with the count in bullet points. Also provide heading identified ingredients in markdown heading"
                            }
                        ]
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
            modelId='amazon.nova-pro-v1:0',
            body=json.dumps(input_prompt),
            contentType='application/json',
            accept='application/json'
        )

        model_response = json.loads(response["body"].read())
        
        # Pretty print the response JSON for debugging
        print(json.dumps(model_response, indent=2))
        
        # Extract ingredients from response
        ingredients = model_response["output"]["message"]["content"][0]["text"]
        
        # Add context about the media type
        media_type = "video" if isinstance(media, str) else "image"
        ingredients = f"### Ingredients (detected from {media_type})\n{ingredients}"
        
        print("\n[Response Content Text]")
        print(ingredients)
        
        return ingredients

    except Exception as e:
        error_msg = f"Error processing {'video' if isinstance(media, str) else 'image'}: {str(e)}"
        print(error_msg)
        return f"### Error\n{error_msg}"

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
                            1. A simple recipe title (keep it bold heading)
                            2. Estimated cooking time and Serving size
                            2. List of ingredients with measurements
                            3. Step-by-step cooking instructions
                            4. Nutritional value (include calories, protein, fat, carbs)"""
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
    if img is None:
        gr.Warning("No image captured!")
        return None, None
    return img, img

def process_media_input(media, text_input=None, progress=gr.Progress()):
    """
    Process either image or video input with progress updates
    """
    try:
        progress(0.1, desc="Starting media processing...")
        
        if media is None:
            gr.Warning("Uh-oh! No media detected!")
            return "### Error\nNo media provided", ""

        # Check if media is a file path (string from video upload)
        if isinstance(media, str):
            is_video = True
        # Check if media is numpy array (from image upload)
        elif isinstance(media, np.ndarray):
            is_video = len(media.shape) == 4
        else:
            return "### Error\nInvalid media format", ""

        progress(0.3, desc="Analyzing media...")
        
        # Process based on input type
        if is_video:
            progress(0.4, desc="Processing video...")
            ingredients = extract_ingredients(media)
        else:
            progress(0.4, desc="Processing image...")
            ingredients = extract_ingredients(media)

        progress(0.6, desc="Generating recipe...")
        
        # Generate recipe if ingredients were successfully extracted
        if not ingredients.startswith("### Error"):
            combined_input = f"{ingredients}\n{text_input}" if text_input else ingredients
            recipe = generate_recipes(combined_input)
            progress(0.9, desc="Complete!")
            return ingredients, recipe
        else:
            progress(0.9, desc="Error occurred")
            return ingredients, "### Error\nCould not generate recipe due to ingredient detection failure"

    except Exception as e:
        progress(1.0, desc="Error occurred")
        return f"### Error\nProcessing failed: {str(e)}", "### Error\nUnable to generate recipe"

def process_video(video, text_input):
    if video is None:
        gr.Warning("No video provided!")
        return "### Error\nNo video to analyze", ""
    
    try:
        # Your video processing logic here
        # You might want to extract frames or process the video in some way
        ingredients = extract_ingredients(video)
        recipe = generate_recipes(ingredients, text_input)
        return ingredients, recipe
    except Exception as e:
        return f"### Error\n{str(e)}", ""

with gr.Blocks(title="FlavorAI", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🍳 FlavorAI")
    gr.Markdown("Upload an image, capture from webcam, or provide ingredients to generate recipes.")
    with gr.Row():
        with gr.Column(scale=1):
            # Tabs for different input methods
            with gr.Tabs():
                with gr.TabItem("Image"):
                    upload_input = gr.Image(
                        type="numpy",
                        label="Upload an Image",
                        sources=["upload"],
                        height=350
                    )
                    
                    upload_analyze_btn = gr.Button("Analyze Uploaded Image", variant="primary")

                # New Video Tab
                with gr.TabItem("Video"):
                    video_upload = gr.Video(
                        label="Upload a Video",
                        sources=["upload"],
                        height=350
                    )
                    video_upload_analyze_btn = gr.Button("Analyze Uploaded Video", variant="primary")
            
            # Text input
            text_input = gr.Textbox(
                label="Additional Ingredients or Preferences",
                placeholder="Enter any additional ingredients or preferences (optional)",
                lines=2
            )
        
        with gr.Column(scale=1):
            ingredients_output = gr.Markdown(label="Detected Ingredients")
            recipe_output = gr.Markdown(label="Generated Recipe")

    # Existing image capture functionality
    def capture_image(img):
        if img is None:
            gr.Warning("No image captured!")
            return None, None
        return img, img

    # New video recording functionality
    def handle_video(video):
        if video is None:
            gr.Warning("No video recorded!")
            return None, None
        return video, video

    # Analysis functionalities
    upload_analyze_btn.click(
        fn=process_media_input,
        inputs=[upload_input, text_input],
        outputs=[ingredients_output, recipe_output],
        api_name="analyze_upload",
        show_progress=True
    )

    # New video analysis functionalities
    video_upload_analyze_btn.click(
        fn=process_media_input,
        inputs=[video_upload, text_input],
        outputs=[ingredients_output, recipe_output],
        api_name="analyze_uploaded_video",
        show_progress=True
    )

if __name__ == "__main__":
    iface.launch()
