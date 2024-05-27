import torch
import time
from torch.utils.data import DataLoader
from dataset.dcase24_dev import get_training_set
from helpers.init import worker_init_fn
from dataset.dcase24_dev_logmel import get_training_set_log

# Function to obtain datasets
def get_datasets():
    dataset1 = get_training_set_log(5, roll=0)
    dataset2 = get_training_set(5, roll=0)
    return dataset1, dataset2

def measure_dataloader_speed(dataloader, num_batches):
    start_time = time.time()
    for i, _ in enumerate(dataloader):
        if i >= num_batches:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    average_time_per_batch = elapsed_time / num_batches
    return average_time_per_batch

def main():
    # Obtain datasets from user-provided script
    dataset1, dataset2 = get_datasets()
    
    # Define dataloaders
    batch_size = 256
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    
    # Measure the speed of dataloaders
    num_batches_to_test = 100
    avg_time_dataloader1 = measure_dataloader_speed(dataloader1, num_batches_to_test)
    avg_time_dataloader2 = measure_dataloader_speed(dataloader2, num_batches_to_test)
    
    print(f"Average loading time per batch for .pt Dataloader: {avg_time_dataloader1:.4f} seconds")
    print(f"Average loading time per batch for .wav to Logmel Dataloader: {avg_time_dataloader2:.4f} seconds")

if __name__ == "__main__":
    main()
