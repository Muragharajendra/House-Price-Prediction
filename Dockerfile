# Use a lightweight python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any are needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY model.pkl pipeline.pkl app.py frontend.py housing.csv input.csv Main.py README.md start.sh ./

# Make the startup script executable
RUN chmod +x start.sh

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run the application using the startup script
CMD ["./start.sh"]
