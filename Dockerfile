# Build stage for React frontend
FROM node:16 as build-stage

WORKDIR /app/frontend

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy frontend files and build
COPY frontend/ ./
RUN npm run build

# Production stage for Python backend
FROM python:3.9

WORKDIR /app

# Copy built React files
COPY --from=build-stage /app/frontend/build /app/frontend/build

# Backend setup
WORKDIR /app/backend
COPY backend/requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY run.py /app/

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PORT=4000

# Expose port
EXPOSE 4000

# Run the application
ENTRYPOINT ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:$PORT"]
