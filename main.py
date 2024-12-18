import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the project directory structure
project_dir = 'shm_algorithms'
data_dir = os.path.join(project_dir, 'data')
src_dir = os.path.join(project_dir, 'src')
results_dir = os.path.join(project_dir, 'results')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(src_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load the dataset from text files
sensor_data = pd.DataFrame()
for file in os.listdir(data_dir):
    if file.endswith('.txt'):
        df = pd.read_csv(os.path.join(data_dir, file), sep='\t')
        sensor_data = pd.concat([sensor_data, df], ignore_index=True)

# Implement the algorithms
from src.retasking import RetaskingAlgorithm
from src.clustering import ClusteringAlgorithm
from src.ftmr import FTMRAlgorithm
from src.rl import RLAlgorithm

retasking = RetaskingAlgorithm(sensor_data)
retasking.dynamic_reassignment()
retasking.evaluate_performance()

clustering = ClusteringAlgorithm(sensor_data)
clustering.cluster_formation()
clustering.fault_detection()
clustering.evaluate_performance()

ftmr = FTMRAlgorithm(sensor_data)
ftmr.redundancy_management()
ftmr.fault_tolerance()
ftmr.evaluate_performance()

rl = RLAlgorithm(sensor_data)
rl.state_action_design()
rl.train_agent()
rl.evaluate_performance()

# Compare the performance metrics
print("Retasking Algorithm:")
print(f"Accuracy: {retasking.accuracy:.2f}")
print(f"False Positive Rate: {retasking.false_positive_rate:.2f}")
print(f"Response Time: {retasking.response_time:.2f} seconds")
print(f"Resource Utilization: {retasking.resource_utilization:.2f}")
print(f"Fault Tolerance: {retasking.fault_tolerance:.2f}")

print("\nClustering Algorithm:")
print(f"Accuracy: {clustering.accuracy:.2f}")
print(f"False Positive Rate: {clustering.false_positive_rate:.2f}")
print(f"Response Time: {clustering.response_time:.2f} seconds")
print(f"Resource Utilization: {clustering.resource_utilization:.2f}")
print(f"Fault Tolerance: {clustering.fault_tolerance:.2f}")

print("\nFTMR Algorithm:")
print(f"Accuracy: {ftmr.accuracy:.2f}")
print(f"False Positive Rate: {ftmr.false_positive_rate:.2f}")
print(f"Response Time: {ftmr.response_time:.2f} seconds")
print(f"Resource Utilization: {ftmr.resource_utilization:.2f}")
print(f"Fault Tolerance: {ftmr.fault_tolerance:.2f}")

print("\nReinforcement Learning Algorithm:")
print(f"Accuracy: {rl.accuracy:.2f}")
print(f"False Positive Rate: {rl.false_positive_rate:.2f}")
print(f"Response Time: {rl.response_time:.2f} seconds")
print(f"Resource Utilization: {rl.resource_utilization:.2f}")
print(f"Fault Tolerance: {rl.fault_tolerance:.2f}")

# Generate plots
plt.figure(figsize=(12, 6))
plt.bar(['Retasking', 'Clustering', 'FTMR', 'RL'],
       [retasking.accuracy, clustering.accuracy, ftmr.accuracy, rl.accuracy])
plt.title('Comparison of Detection Accuracy')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))

plt.figure(figsize=(12, 6))
plt.bar(['Retasking', 'Clustering', 'FTMR', 'RL'],
       [retasking.false_positive_rate, clustering.false_positive_rate, ftmr.false_positive_rate, rl.false_positive_rate])
plt.title('Comparison of False Positive Rates')
plt.xlabel('Algorithm')
plt.ylabel('False Positive Rate')
plt.savefig(os.path.join(results_dir, 'false_positive_comparison.png'))

plt.figure(figsize=(12, 6))
plt.bar(['Retasking', 'Clustering', 'FTMR', 'RL'],
       [retasking.response_time, clustering.response_time, ftmr.response_time, rl.response_time])
plt.title('Comparison of Response Times')
plt.xlabel('Algorithm')
plt.ylabel('Response Time (seconds)')
plt.savefig(os.path.join(results_dir, 'response_time_comparison.png'))

plt.figure(figsize=(12, 6))
plt.bar(['Retasking', 'Clustering', 'FTMR', 'RL'],
       [retasking.resource_utilization, clustering.resource_utilization, ftmr.resource_utilization, rl.resource_utilization])
plt.title('Comparison of Resource Utilization')
plt.xlabel('Algorithm')
plt.ylabel('Resource Utilization')
plt.savefig(os.path.join(results_dir, 'resource_utilization_comparison.png'))

plt.figure(figsize=(12, 6))
plt.bar(['Retasking', 'Clustering', 'FTMR', 'RL'],
       [retasking.fault_tolerance, clustering.fault_tolerance, ftmr.fault_tolerance, rl.fault_tolerance])
plt.title('Comparison of Fault Tolerance')
plt.xlabel('Algorithm')
plt.ylabel('Fault Tolerance')
plt.savefig(os.path.join(results_dir, 'fault_tolerance_comparison.png'))