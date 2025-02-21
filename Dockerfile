# Build stage for React frontend
FROM node:14 as build-stage

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Production stage for Python backend
FROM python:3.9

WORKDIR /app

# Copy built React files
COPY --from=build-stage /app/frontend/build /app/frontend/build

# Copy backend files
COPY backend/ /app/backend/
COPY run.py /app/

# Install Python dependencies
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "run:app"]