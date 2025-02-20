# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install Node.js and npm
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Verify Node.js and npm installation
RUN node --version && npm --version

# Copy the current directory contents into the container at /app
COPY . /app/

# Install backend dependencies
RUN cd backend && pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies and build
WORKDIR /app/frontend
RUN npm ci
RUN npm run build

# Change back to the main directory
WORKDIR /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "backend/app.py"]