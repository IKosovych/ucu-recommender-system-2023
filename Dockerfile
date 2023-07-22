# Use the official Python image as a base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the app.py file and other required files
COPY app.py .
COPY evaluation.py evaluation.py
COPY models/ models/
COPY data/ data/
COPY data_uploader/ data_uploader/
COPY requirements.txt requirements.txt

# Install the required dependencies
RUN pip install -r requirements.txt

# Expose the port that Streamlit app is running on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
