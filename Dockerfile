# Build stage for React frontend
FROM node:16 as build-stage

WORKDIR /proyecto-chatbot-becas/frontend

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy frontend files and build
COPY frontend/ ./
RUN npm run build

# Production stage for Python backend
FROM python:3.9

WORKDIR /proyecto-chatbot-becas

# Copy built React files
COPY --from=build-stage /proyecto-chatbot-becas/frontend/build /proyecto-chatbot-becas/frontend/build

# Backend setup
WORKDIR /proyecto-chatbot-becas/backend
COPY backend/requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /proyecto-chatbot-becas/backend
#COPY app.py /proyecto-chatbot-becas/backend/app.py

# Set environment variables
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV PORT=4000
ENV MYSQL_HOST=${MYSQL_HOST}
ENV MYSQL_USER=${MYSQL_USER}
ENV MYSQL_PASSWORD=${MYSQL_PASSWORD}
ENV MYSQL_DB=${MYSQL_DB}
ENV MYSQL_PORT=${MYSQL_PORT:-3306}

# Expose port
EXPOSE 4000

# Run the application
ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "app:app"]

