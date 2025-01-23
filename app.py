from PIL import Image
import gradio as gr
import boto3
import json
import base64
import io
import numpy as np

# Initialize Boto3 client for Bedrock
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

def extract_ingredients(media):
    try:
        if isinstance(media, str):  # Handle video input
            with open(media, "rb") as video_file:
                binary_data = video_file.read()
                base64_string = base64.b64encode(binary_data).decode("utf-8")
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
                                "text": """You are an expert media analyst and a professional in identifying food ingredients from visual content.
                                    List only the exact ingredients visible in the video. Be specific in naming the ingredients (e.g., "red pepper" instead of "bell pepper").
                                    Exclude non-edible items. Provide the ingredients and their count as bullet points."""
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 2048,
                    "top_p": 0.95,
                    "top_k": 250,
                    "temperature": 1
                }
            }
        else:  # Handle image input
            pil_image = Image.fromarray(media)
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
                                "text": """You are an expert media analyst and a professional in identifying food ingredients from visual content.
                                    List only the exact ingredients visible in the image. Be specific in naming the ingredients (e.g., "red pepper" instead of "bell pepper").
                                    Exclude non-edible items. Provide the ingredients and their count as bullet points."""
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 2048,
                    "top_p": 0.95,
                    "top_k": 250,
                    "temperature": 0.7
                }
            }

        # Call Bedrock model
        response = bedrock_client.invoke_model(
            modelId='amazon.nova-pro-v1:0',
            body=json.dumps(input_prompt),
            contentType='application/json',
            accept='application/json'
        )

        model_response = json.loads(response["body"].read())
        try:
            ingredients = model_response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as e:
            return f"### Error\nUnexpected model response format: {str(e)}"
        
        media_type = "video" if isinstance(media, str) else "image"
        return f"### Ingredients (detected from {media_type})\n{ingredients}"

    except Exception as e:
        return f"### Error\nError processing media: {str(e)}"

def generate_recipes(combined_input):
    try:
        input_prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"""Generate a recipe based on the following ingredients:
                            Available ingredients: {combined_input}
                            
                            Please provide:
                            1. A simple recipe title (bold heading)
                            2. Estimated cooking time and serving size
                            3. Nutritional value (calories, protein, fat, carbs)
                            4. List of ingredients with measurements
                            5. Step-by-step cooking instructions"""
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
            modelId='amazon.nova-micro-v1:0',
            body=json.dumps(input_prompt),
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        try:
            recipe = response_body["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as e:
            return f"### Error\nUnexpected model response format: {str(e)}"
        
        return recipe

    except Exception as e:
        return f"### Error\nError generating recipe: {str(e)}"

def process_media_input(media, text_input=None, ingredients_state=None, progress=gr.Progress()):
    try:
        progress(0.1, desc="Starting media processing...")

        # Validate media input or use cached state
        if media is None and not ingredients_state:
            return "### Error\nNo media or cached data provided", ""

        if not isinstance(media, (str, np.ndarray)) and media is not None:
            return "### Error\nUnsupported media format", ""

        # Use cached ingredients if available
        if ingredients_state:
            ingredients = ingredients_state
            progress(0.3, desc="Using cached ingredients...")
        else:
            ingredients = extract_ingredients(media)
            if ingredients.startswith("### Error"):
                return ingredients, ""
        
        progress(0.6, desc="Generating recipe...")
        combined_input = f"{ingredients}\n{text_input}" if text_input else ingredients
        recipe = generate_recipes(combined_input)
        
        progress(0.9, desc="Complete!")
        return ingredients, recipe

    except Exception as e:
        return f"### Error\nProcessing failed: {str(e)}", "### Error\nUnable to generate recipe"

custom_css = """
.gradio-container {
    background: linear-gradient(#135deg, #f3f4f6, #e2e8f0);
    padding: 10px;
    margin: 0;
    max-width: 800px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    border-radius: 6px;
}
.gr-row {
    margin-bottom: 20px; /* Add spacing between rows */
}
.button {
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
}
.button:hover {
    background-color: #4a90e2; /* Lighter hover effect */
}
.body {
    font-family: 'Roboto', sans-serif;
}
.gr-markdown {
    margin: 5px 0 
}
"""

with gr.Blocks(css=custom_css, title="FlavorAI", theme=gr.themes.Soft(primary_hue="indigo", spacing_size="sm")) as iface:
    ingredients_state = gr.State()
    gr.Markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center; padding: 10px;'>
            <div style='flex: 1;'>
                <img src='https://i.imghippo.com/files/kPy5369H.png' style='width: 150px; height: 50px; object-fit: contain;'>
            </div>
            <center><div style='flex: 2; text-align: center;'>
                <img src="https://i.imghippo.com/files/eUe1202ed.png" alt="" border="0" style='width: 300px; height: 100px; object-fit: contain;'>
            </div></center>
            <div style='flex: 1;'></div>
        </div>
        <center><p style='color: #666666; font-size: 16px; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.1); margin: 5px 0;'>Snap it, upload it, and let AI do the cooking math!</p></center>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🖼️ Image"):
                    upload_input = gr.Image(type="numpy", label="Upload an Image", sources=["upload"], height=350)
                    text_input = gr.Textbox(label="Additional Preferences (optional)", placeholder="Enter additional preferences...")
                    analyze_btn = gr.Button("Analyze Uploaded Image")
                with gr.TabItem("🎥 Video"):
                    video_upload = gr.Video(label="Upload a Video", sources=["upload"], height=350)
                    text_input = gr.Textbox(label="Additional Preferences (optional)", placeholder="Enter additional preferences...")
                    analyze_video_btn = gr.Button("Analyze Uploaded Video")
        with gr.Column(scale=1):
            ingredients_output = gr.Markdown(label="Detected Ingredients")
            recipe_output = gr.Markdown(label="Generated Recipe")

    analyze_btn.click(
        fn=process_media_input,
        inputs=[upload_input, text_input, ingredients_state],
        outputs=[ingredients_output, recipe_output],
        show_progress=True
    )
    
    analyze_btn.click(
        fn=lambda ingredients: ingredients,
        inputs=[ingredients_output],
        outputs=[ingredients_state]
    )

    analyze_video_btn.click(
        fn=process_media_input,
        inputs=[video_upload, text_input, ingredients_state],
        outputs=[ingredients_output, recipe_output],
        show_progress=True
    )
    
if __name__ == "__main__":
    iface.launch()
