# Stage 1: Build React frontend
FROM node:16 as build-stage

WORKDIR /proyecto-chatbot-becas/frontend

# Copiar archivos de configuración
COPY frontend/package*.json ./ 

# Instalar dependencias
RUN npm ci --legacy-peer-deps

# Copiar los archivos del frontend y construir la aplicación
COPY frontend/ . 
RUN npm run build

# Stage 2: Backend (Flask)
FROM python:3.9 as backend-stage

WORKDIR /proyecto-chatbot-becas

# Copiar y instalar dependencias del backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del backend
COPY backend/ /proyecto-chatbot-becas/backend

# Configurar variables de entorno
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV PORT=4000

# Configurar variables para MySQL
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

# Exponer el puerto 4000
EXPOSE 4000

# Iniciar Flask con Gunicorn
CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:4000", "app:app"]

# Stage 3: Nginx to serve the frontend
FROM nginx:alpine as frontend-stage

# Copiar los archivos de la build de React a la carpeta de Nginx
COPY --from=build-stage /proyecto-chatbot-becas/frontend/build /usr/share/nginx/html

# Exponer el puerto 80
EXPOSE 80

# Iniciar Nginx con configuración predeterminada
CMD ["nginx", "-g", "daemon off;"]
