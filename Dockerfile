# Imagen base
FROM python:3.10-slim

# Carpeta de trabajo
WORKDIR /app

# Actualizar pip
RUN pip install --upgrade pip

# Copiar requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto 5000
EXPOSE 5000

# Usar Gunicorn para producción
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
