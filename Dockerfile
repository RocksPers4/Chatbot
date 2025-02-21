# Build stage for React frontend
FROM node:16 as build-stage

WORKDIR /app/frontend

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies with legacy-peer-deps to avoid conflicts
RUN npm ci --legacy-peer-deps

# Install missing Babel dependency (fixes build error)
RUN npm install --save-dev @babel/plugin-proposal-private-property-in-object

# Copy frontend files
COPY frontend/ ./

# Build the React app
RUN npm run build

# Production stage for Python backend
FROM python:3.9

WORKDIR /app

# Copy built React files
COPY --from=build-stage /app/frontend/build /app/frontend/build

# Copy backend files
COPY backend/ /app/backend/
COPY run.py /app/

# Install Python dependencies (fix incorrect path)
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PORT=5000

# Expose the correct port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-5000}", "run:app"]
