# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt before copying other files
COPY requirements.txt /app/

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the Flask app port
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]
