# 1. Base Image: Lightweight Python environment
# Alpine Linux use kar rahe hain, jo chota hota hai
FROM python:3.10-slim

# Environment variable set karo
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# 2. Files Copy Karo
# requirements.txt ko pehle copy karte hain taaki caching fast ho
COPY requirements.txt .

# 3. Dependencies Install Karo
# Flask aur gunicorn ko bhi add karna padega agar requirements.txt mein nahi hain
# Gunicorn is needed for production server
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn flask

# 4. MLflow Tracking Setup (Required to load model from MLflow)
# Hamein MLflow server se model load karna hai, jiske liye ho sakta hai ki aapka 
# MLflow Tracking URI private ho. Lekin is case mein hum local model artifacts 
# (mlruns folder) ko use karenge jo CI/CD ne banaya tha.

# CI/CD se aaye hue artifacts ko copy karo (jo git mein aana chahiye)
COPY mlruns/ $APP_HOME/mlruns/

# app.py script ko copy karo
COPY app.py $APP_HOME

# 5. Container Run Command
# Port 5000 ko expose karo (jahan Flask server run hoga)
EXPOSE 5000

# Gunicorn use karke production-ready server run karo
# Workers 2 ya 4 use kar sakte hain
CMD exec gunicorn --bind 0.0.0.0:5000 --workers 4 app:app