FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY week2/api.py ./api.py
COPY models/asteroid_model.pkl ./models/asteroid_model.pkl
COPY models/asteroid_features.pkl ./models/asteroid_features.pkl
COPY data/asteroids.csv ./data/asteroids.csv

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
