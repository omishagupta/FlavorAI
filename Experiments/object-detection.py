import boto3

def detect_objects(image_path=None, bucket=None, key=None):
    """
    Detect objects in an image using Amazon Rekognition.
    You can provide an image either from a local file or an S3 bucket.
    
    Args:
        image_path (str): Path to a local image file.
        bucket (str): Name of the S3 bucket (if using S3).
        key (str): S3 object key (if using S3).
    
    Returns:
        list: Detected objects with confidence levels.
    """
    # Initialize the Rekognition client
    rekognition_client = boto3.client('rekognition')

    # Prepare the image data
    if image_path:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        image_data = {'Bytes': image_bytes}
    elif bucket and key:
        image_data = {'S3Object': {'Bucket': bucket, 'Name': key}}
    else:
        raise ValueError("Provide either a local image_path or bucket/key for S3.")

    # Call Rekognition's detect_labels API
    response = rekognition_client.detect_labels(
        Image=image_data,
        MaxLabels=10,  # Maximum number of labels to return
        MinConfidence=50  # Minimum confidence level for a label to be included
    )

    # Extract and return detected labels
    labels = response.get('Labels', [])
    detected_objects = [
        {'Name': label['Name'], 'Confidence': label['Confidence']}
        for label in labels
    ]
    return detected_objects


if __name__ == '__main__':
    # Example usage:
    # Detect objects from a local file
    image_path = 'fridge-basket.jpg'  # Replace with your local image path

    # # Detect objects from an S3 bucket
    # bucket_name = '~/Downloads/pots.jpg'  # Replace with your S3 bucket name
    # object_key = 'path/to/image/in/s3.jpg'  # Replace with your S3 object key

    # Uncomment one of the following lines to use:
    try:
        # For local file
        result = detect_objects(image_path=image_path)

        # OR for S3
        # result = detect_objects(bucket=bucket_name, key=object_key)

        # Print detected objects
        for obj in result:
            print(f"Detected: {obj['Name']} (Confidence: {obj['Confidence']:.2f}%)")

    except Exception as e:
        print(f"Error: {e}")
