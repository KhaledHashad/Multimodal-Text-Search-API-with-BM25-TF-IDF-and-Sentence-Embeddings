<<<<<<< HEAD
# Use Python base image
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk tokenizer
RUN python -m nltk.downloader punkt

# Copy app code
COPY app.py .

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
=======
# Use Python base image
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk tokenizer
RUN python -m nltk.downloader punkt

# Copy app code
COPY app.py .

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
>>>>>>> 031a706 (Initial working search API with Docker)
