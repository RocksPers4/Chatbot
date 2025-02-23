# Build stage for React frontend
FROM node:16 as build-stage

WORKDIR /proyecto-chatbot-becas/frontend

# Copiar archivos de configuración
COPY frontend/package*.json ./

# Instalar dependencias
RUN npm ci --legacy-peer-deps

# Copiar los archivos del frontend y construir la aplicación
COPY frontend/ .
RUN npm run build

# Production stage: Usar Nginx para servir la aplicación
FROM nginx:alpine

# Copiar los archivos construidos al servidor web Nginx
COPY --from=build-stage /frontend/build /usr/share/nginx/html

# Exponer el puerto 80
EXPOSE 80

# Comando de inicio para Nginx
CMD ["nginx", "-g", "daemon off;"]
