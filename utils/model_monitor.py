import csv
from datetime import datetime
import os
import pandas as pd

LOG_PATH = "logs/model_logs.csv"

def log_model_run(model_name, metric, value):
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        
        # Check if file exists to write headers
        file_exists = os.path.exists(LOG_PATH)
        
        # Create or append to the log file
        with open(LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write headers if file is new
            if not file_exists:
                headers = ['Timestamp', 'Model', 'Metric', 'Value']
                writer.writerow(headers)
            
            # Format the timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the model run data
            writer.writerow([timestamp, model_name, metric, value])
            
        # Verify the file was written correctly
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            if not all(col in df.columns for col in ['Timestamp', 'Model', 'Metric', 'Value']):
                print(f"Warning: Log file columns don't match expected format. Found: {df.columns.tolist()}")
                
    except Exception as e:
        print(f"Error logging model run: {str(e)}")
        raise 
