FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_md \
 && python -c "import nltk; nltk.download('stopwords', download_dir='/home/user/.nltk_data'); nltk.download('wordnet', download_dir='/home/user/.nltk_data'); nltk.download('punkt', download_dir='/home/user/.nltk_data')" \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m user

# Copy application code
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]