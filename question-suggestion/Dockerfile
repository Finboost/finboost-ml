# Use the official Python image as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the Flask app
EXPOSE 8080

# Define environment variable for Flask app
ENV FLASK_APP=app/main.py

# Install Gunicorn
RUN pip install gunicorn

# Run Gunicorn server when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]
