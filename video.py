# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import boto3
import json
import gradio as gr

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
)

MODEL_ID = "us.amazon.nova-lite-v1:0"
# Open the image you'd like to use and encode it as a Base64 string.
def extract_ingredients(media):
    # Encode the image as a Base64 string.
    if media is None:
        return "No image provided"
    else:
        print("Image provided")
        
    with open(media, "rb") as video_file:
        binary_data = video_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")
    # Define your system prompt(s).
    system_list= [
        {
            "text": "You are an expert media analyst. When the user provides you with a video, provide 3 potential video titles"
        }
    ]
    # Define a "user" message including both the image and a text prompt.
    message_list = [
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
                    "text": "Provide video titles for this clip."
                },
            ],
        }
    ]
    # Configure the inference parameters.
    inf_params = {"max_new_tokens": 300, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    # Invoke the model and extract the response body.
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())
    # Pretty print the response JSON.
    print("[Full Response]")
    print(json.dumps(model_response, indent=2))
    # Print the text content for easy readability.
    content_text = model_response["output"]["message"]["content"][0]["text"]
    print("\n[Response Content Text]")
    return content_text

iface = gr.Interface(
    fn=extract_ingredients,
    inputs = gr.Video(
                        label="Upload a Video",
                        sources=["upload", "webcam"],
                        height=350 ),
    outputs="text",
    title="FlavorAI",
    description="Upload an image of food, and this tool will analyze it to extract the list of ingredients."
)


if __name__ == "__main__":
    iface.launch()
