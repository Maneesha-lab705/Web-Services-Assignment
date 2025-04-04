# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies (if applicable)
RUN pip install -r requirements.txt

# Command to run the service
CMD ["python", "app.py"]
