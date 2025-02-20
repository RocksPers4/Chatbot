# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install Node.js and npm
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Copy the current directory contents into the container at /app
COPY . /app/

# Install backend dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip cache purge

# Install frontend dependencies and build
WORKDIR /app/frontend
RUN npm ci --legacy-peer-deps
RUN npm run build --max-old-space-size=256

# Change back to the main directory
WORKDIR /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production
ENV PORT=8080

# Run app.py when the container launches
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} backend.app:app"]