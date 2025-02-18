#!/bin/bash

# Build Docker image
echo "Building Docker image..."
docker build -t fraud-detection-model .

# Stop existing container if running
echo "Stopping existing container..."
docker stop fraud-detection-model || true
docker rm fraud-detection-model || true

# Run new container
echo "Starting new container..."
docker run -d \
    --name fraud-detection-model \
    -p 5000:5000 \
    -v $(pwd)/logs:/app/logs \
    fraud-detection-model

echo "Deployment complete!" 