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

# Run the application
ENTRYPOINT ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:4000", "app:app"]
