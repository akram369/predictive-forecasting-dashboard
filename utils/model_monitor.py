import csv
from datetime import datetime
import os
import pandas as pd

LOG_PATH = "logs/model_logs.csv"

def log_model_run(model_name, metric, value):
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        
        # Initialize or load existing data
        if os.path.exists(LOG_PATH):
            try:
                df = pd.read_csv(LOG_PATH)
                next_index = len(df)
            except:
                df = pd.DataFrame(columns=['index', 'Model', 'Value', 'Timestamp'])
                next_index = 0
        else:
            df = pd.DataFrame(columns=['index', 'Model', 'Value', 'Timestamp'])
            next_index = 0
        
        # Add new row
        new_row = pd.DataFrame({
            'index': [next_index],
            'Model': [model_name],
            'Value': [value],
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        # Combine and save
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(LOG_PATH, index=False)
            
    except Exception as e:
        print(f"Error logging model run: {str(e)}")
        raise 
