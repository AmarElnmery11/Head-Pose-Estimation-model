# Use an official Python runtime as a parent image
FROM python:latest
# Set the working directory in the container
WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .
# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory tpip freeze > requirements.txto the working directory
COPY . .
# Make port 8008 available to the world outside this container
EXPOSE 8008
# This CMD starts the uvicorn server directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]