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
                headers = ['index', 'Model', 'Value', 'Timestamp']
                writer.writerow(headers)
            
            # Get the next index
            next_index = 0
            if file_exists:
                try:
                    df = pd.read_csv(LOG_PATH)
                    next_index = len(df)
                except:
                    pass
            
            # Format the timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the model run data
            writer.writerow([next_index, model_name, value, timestamp])
            
    except Exception as e:
        print(f"Error logging model run: {str(e)}")
        raise 
