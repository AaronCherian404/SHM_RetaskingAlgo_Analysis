import os
import numpy as np
import pandas as pd
from typing import List, Dict

class HydraulicSystemDataLoader:
    def __init__(self, dataset_path: str):
        """
        Initialize the data loader for Hydraulic System dataset
        
        :param dataset_path: Path to the directory containing dataset files
        """
        self.dataset_path = dataset_path
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = [
            'Cooler_Temperature', 
            'Valve_Opening', 
            'Pump_Displacement', 
            'Hydraulic_Oil_Temperature', 
            'Cooling_Efficiency', 
            'Hydraulic_Oil_Level'
        ]
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from text files in the dataset directory
        
        :return: Pandas DataFrame with combined data
        """
        data_files = [
            f for f in os.listdir(self.dataset_path) 
            if f.endswith('.txt') and 'profile' in f.lower()
        ]
        
        all_data = []
        
        for file in data_files:
            file_path = os.path.join(self.dataset_path, file)
            
            # Read the text file
            file_data = pd.read_csv(
                file_path, 
                sep='\t', 
                header=None, 
                names=self.feature_columns + ['Condition']
            )
            
            all_data.append(file_data)
        
        # Combine all data files
        self.raw_data = pd.concat(all_data, ignore_index=True)
        return self.raw_data
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the raw data for SHM algorithms
        
        :return: Preprocessed DataFrame
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        # Create a copy for preprocessing
        processed_df = self.raw_data.copy()
        
        # Handle missing values
        processed_df.fillna(processed_df.mean(), inplace=True)
        
        # Normalize numerical columns
        for col in self.feature_columns:
            processed_df[col] = (processed_df[col] - processed_df[col].mean()) / processed_df[col].std()
        
        # Create binary anomaly label (for demonstration)
        processed_df['is_anomaly'] = (processed_df['Condition'] != 0).astype(int)
        
        self.processed_data = processed_df
        return self.processed_data
    
    def get_anomaly_ground_truth(self) -> np.ndarray:
        """
        Get ground truth anomaly labels
        
        :return: NumPy array of binary anomaly labels
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        return self.processed_data['is_anomaly'].values