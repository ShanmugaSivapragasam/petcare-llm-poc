# Use a lightweight Python base image
FROM python:3.10.13-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install ctransformers --no-binary ctransformers



# Copy the application code
COPY . .

# Expose the port (if your application runs on a specific port)
 EXPOSE 7861

# Set the entry point for the container
CMD ["python", "app.py"]