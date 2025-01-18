# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gradio runs on
EXPOSE 7860

# Set environment variable to allow Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the application
CMD ["python", "app.py"]
