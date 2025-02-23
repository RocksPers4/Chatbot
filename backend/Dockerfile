# Stage 1: Base image for Python backend
FROM python:3.9 AS backend-stage

WORKDIR /proyecto-chatbot-becas/backend

# Copy requirements.txt to install dependencies
COPY backend/requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code
COPY backend/ /proyecto-chatbot-becas/backend

# Set environment variables
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV PORT=4000
ARG MYSQLHOST
ARG MYSQLUSER
ARG MYSQLPASSWORD
ARG MYSQLDATABASE
ARG MYSQLPORT

ENV MYSQL_HOST=$MYSQLHOST
ENV MYSQL_USER=$MYSQLUSER
ENV MYSQL_PASSWORD=$MYSQLPASSWORD
ENV MYSQL_DB=$MYSQLDATABASE
ENV MYSQL_PORT=$MYSQLPORT

# Expose port
EXPOSE 4000

# Run the application using Gunicorn
ENTRYPOINT ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:4000", "app:app"]
