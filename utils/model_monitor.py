import csv
from datetime import datetime
import os

LOG_PATH = "logs/model_logs.csv"

def log_model_run(model_name, metric, value):
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    # Check if file exists to write headers
    file_exists = os.path.exists(LOG_PATH)
    
    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Model', 'Metric', 'Value'])
        # Write the model run data
        writer.writerow([datetime.now(), model_name, metric, value]) 
