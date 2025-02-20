# Usa una imagen oficial de Python
FROM python:3.9

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar Node.js y npm
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Instalar npm globalmente con permisos de root
RUN npm cache clean --force
RUN npm install -g npm@latest --unsafe-perm=true --verbose

# Copiar los archivos del proyecto al contenedor
COPY . /app/

# Instalar las dependencias de backend
RUN pip install --no-cache-dir -r backend/requirements.txt

# Instalar dependencias de frontend y construir
WORKDIR /app/frontend
RUN npm install --legacy-peer-deps
RUN npm install --only=dev

# Ejecutar la construcción del frontend
RUN npm run build

# Cambiar de nuevo al directorio principal
WORKDIR /app

# Exponer el puerto 8080 (asegúrate de que coincida con lo que Railway necesita)
EXPOSE 8080

# Definir las variables de entorno para Flask
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production
ENV PORT=8080

# Usar gunicorn para ejecutar la app en producción
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-8080}", "backend.app:app"]

# Limpiar la caché de pip después de la instalación de dependencias
RUN pip cache purge
