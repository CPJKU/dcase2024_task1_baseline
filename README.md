# DCASE2024 - Task 1 - Baseline Systems

Contact: **Florian Schmid** (florian.schmid@jku.at), *Johannes Kepler University Linz*



## Data-Efficient Low-Complexity Acoustic Scene Classification Challenge

For a detailed description of the challenge and this task visit the [DCASE website](https://dcase.community/challenge2024/).

Acoustic scene classification aims to automatically categorize audio recordings into specific environmental sound scenes, such as "metro station," "urban park," or "public square." Previous editions of the acoustic scene classification (ASC) task have focused on limited computational resources and diverse recording conditions, reflecting typical challenges faced when developing ASC models for embedded systems.

This year, participants are additionally encouraged to tackle another problematic condition, namely the limited availability of labeled training data. To this end, the ranking will be based on the number of labeled examples used for training and the system's performance on a test set with diverse recording conditions. Participants will train their system on predefined training sets with a varying number of items. External resources, such as data sets and pre-trained models not specific to ASC, are allowed after the approval of the task organizers and do not count towards the labeled training data count used in the ranking.

Additionally, the focus on low-complexity models is preserved by restricting the model size to 128 kB and the number of multiply-accumulate operations for a one-second audio clip to 30 million.

## Baseline System

This repository contains the code for the baseline system of the DCASE 2024 Challenge Task 1.

* The training loop is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). 
* Logging is implemented using [Weights and Biases](https://wandb.ai/site). 
* The neural network architecture is a simplified version of [CP-Mobile](https://dcase.community/documents/workshop2023/proceedings/DCASE2023Workshop_Schmid_1.pdf), the architecture used in the top-ranked system of [Task 1 in the DCASE 2023 challenge](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification-results).
* The model has 61,148 parameters and 29.42 million MACs. These numbers are counted using [NeSsi](https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py). For inference, the model's parameters are converted to 16-bit floats to meet the memory complexity constraint of 128 kB for model parameters.
* The baseline implements simple data augmentation mechanisms: Time rolling of the waveform and masking of frequency bins and time frames.
* To enhance the generalization across different recording devices, the baseline implements [Frequency-MixStyle](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Schmid_27.pdf). 


## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n d24_t1 python=3.10
conda activate d24_t1
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip3 install torch torchvision torchaudio
```

4. Install requirements:

```
pip3 install -r requirements.txt
```

5. Download and extract the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421).

You should end up with a directory that contains, among other files, the following:
* A directory *audio* containg 230,350 audio files in *wav* format
* A file *meta.csv* that contains 230,350 rows with columns *filename*, *scene label*, *identifier* and *source label*

6. Specify the location of the dataset directory in the variable *dataset_dir* in file [dataset/dcase24.py](dataset/dcase24.py).
7. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.
8. The training procedure can be started by running the following command:
```
python run_training.py --subset=100
```

The subset parameter can be set to one of the values 5, 10, 25, 50, and 100. Changing this parameters, the baseline
can be trained on the available development-train splits. The code will automatically download the correct *split{x}.csv* and
the *test.csv* files from [this location](https://github.com/CPJKU/dcase2024_task1_baseline/releases/tag/files). The downloaded files are placed in a folder *split_setup* (created when starting the training procedure).

##  Data Splits

A detailed description on the data splits (different development-train splits, development-test split and evaluation split) 
is available on the [task description page](https://dcase.community/challenge2024/).

In short: we provide 5 splits of the development-train dataset that contain 5%, 10%, 25%, 50% and 100% of the samples of the full development-train set (corresponds to train set used in DCASE 2023 Task 1).
These are the most important properties about the splits:
* Smaller splits are subsets of larger ones, e.g., all samples of the 5% split are included in the 10% split and all samples of the 25% split are included in the 50% split.
* In all splits the acoustic scenes are approximately uniformly distributed.
* The distribution of recording devices is similar in all splits. 
* A 10-second recording segment is either fully included, or not at all included in a split.

**The system must only be trained on the allowed development-train split (filenames specified in split{x}.csv files) and the explicitly allowed external recources.**

You can suggest additional external resources by writing an email to florian.schmid@jku.at

## Baseline Complexity

The Baseline system has a complexity of 61,148 parameters and 29,419,156 MACs. The table below lists how the parameters
and MACs are distributed across the different layers in the network.

**According to the challenge rules the following complexity limits apply**:
* max memory for model parameters: 128 kB (Kilobyte)
* max number of MACs for inference of a 1-second audio snippet: 30 MMACs (million MACs)

Model parameters of the baseline must therefore be converted to 16-bit precision before inference of the test/evaluation set to stick to the complexity limits (61,148 * 16 bits = 61,148 * 2 B = 122,296 B <= 128 kB).

In previous years of the challenge, top-ranked teams used a technique called **quantization** that converts model paramters to 8-bit precision. In this case,
the maximum number of allowed parameters would be 128,000.


| **Description**       | **Layer**                        | **Input Shape** | **Params** | **MACs**  |
|-----------------------|----------------------------------|-----------------|------------|-----------|
| in_c[0]               | Conv2dNormActivation             | [1, 1, 256, 65] | 88         | 304,144   |
| in_c[1]               | Conv2dNormActivation             | [1, 8, 128, 33] | 2,368      | 2,506,816 |
| stages[0].b1.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b1.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 626,816   |
| stages[0].b1.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 17] | 2,112      | 2,228,288 |
| stages[0].b2.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b2.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 626,816   |
| stages[0].b2.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 17] | 2,112      | 2,228,288 |
| stages[0].b3.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b3.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 331,904   |
| stages[0].b3.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 9]  | 2,112      | 1,179,712 |
| stages[1].b4.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 9]  | 2,176      | 1,179,776 |
| stages[1].b4.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 9]  | 704        | 166,016   |
| stages[1].b4.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 32, 9]  | 3,696      | 1,032,304 |
| stages[1].b5.block[0] | Conv2dNormActivation (pointwise) | [1, 56, 32, 9]  | 6,960      | 1,935,600 |
| stages[1].b5.block[1] | Conv2dNormActivation (depthwise) | [1, 120, 32, 9] | 1,320      | 311,280   |
| stages[1].b5.block[2] | Conv2dNormActivation (pointwise) | [1, 120, 32, 9] | 6,832      | 1,935,472 |
| stages[2].b6.block[0] | Conv2dNormActivation (pointwise) | [1, 56, 32, 9]  | 6,960      | 1,935,600 |
| stages[2].b6.block[1] | Conv2dNormActivation (depthwise) | [1, 120, 32, 9] | 1,320      | 311,280   |
| stages[2].b6.block[2] | Conv2dNormActivation (pointwise) | [1, 120, 32, 9] | 12,688     | 3,594,448 |
| ff_list[0]            | Conv2d                           | [1, 104, 32, 9] | 1,040      | 299,520   |
| ff_list[1]            | BatchNorm2d                      | [1, 10, 32, 9]  | 20         | 20        |
| ff_list[2]            | AdaptiveAvgPool2d                | [1, 10, 32, 9]  | -          | -         |
| **Sum**               | -                                | -               | **61,148**     | **29,419,156** |

To give an example on how MACs and parameters are calculated, let's look in detail into the module **stages[0].b3.block[1]**.
It consists of a conv2d, a batch norm, and a ReLU activation function. 

**Parameters**: The conv2d Parameters are calculated as *input_channels * output_channels * kernel_size * kernel_size*, resulting in 
1 * 64 * 3 * 3 = 576 parametes. Note that *input_channels=1* since it is a depth-wise convolution with 64 groups. The batch norm adds 64 * 2 = 128 parameters
on top, resulting in a total of 704 parameters for this *Conv2dNormActivation* module.

**MACs**: The MACs of the conv2d are calculated as *input_channels * output_channels * kernel_size * kernel_size * output_frequency_bands * output_time_frames*, resulting in 1 * 64 * 3 * 3 * 64 * 9 = 331,776 MACs.   
Note that *input_channels=1* since it is a depth-wise convolution with 64 groups. The batch norm adds 128 MACs
on top, resulting in a total of 331,904 MACs for this *Conv2dNormActivation* module.

## Baseline Results

The primary evaluation metric for the DCASE 2024 challenge Task 1 is **Macro Average Accuracy** (class-wise averaged accuracy). For
exact details on how the submissions are ranked consult the [official task description](https://dcase.community/challenge2024/). 

The two tables below list the Macro Average Accuracy, class-wise accuracies and device-wise accuracies for the baseline system
on all development-train splits (**5%**, **10%**, **25%**, **50%**, **100%**). The results are averaged over 5 runs for each split.
You should obtain similar results when running the baseline system. 



### Class-wise results

|   **Split** | **Airport**   | **Bus**   | **Metro**   | **Metro Station**   | **Park**   | **Public Square**   | **Shopping Mall**   | **Street Pedestrian**   | **Street Traffic**   | **Tram**   | **Macro Average Accuracy**   |
|---------:|:-------------------|:---------------|:-----------------|:-------------------------|:----------------|:-------------------------|:-------------------------|:-----------------------------|:--------------------------|:----------------|:---------------------|
|        **5%** | 34.77            | 45.21        | 30.79          | 40.03                  | 62.06         | 22.28                  | 52.07                  | 31.32                      | 70.23                   | 35.20         | 42.40 ± 0.42              |
|       **10%** | 38.50            | 47.99        | 36.93          | 43.71                  | 65.43         | 27.05                  | 52.46                  | 31.82                      | 72.64                   | 36.41         | 45.29 ± 1.01             |
|       **25%** | 41.81            | 61.19        | 38.88          | 40.84                  | 69.74         | 33.54                  | 58.84                  | 30.31                      | 75.93                   | 51.77         | 50.29 ± 0.87            |
|       **50%** | 41.51            | 63.23        | 43.37          | 48.71                  | 72.55         | 34.25                  | 60.09                  | 37.26                      | 79.71                   | 51.16         | 53.19 ± 0.68            |
|      **100%** | 46.45            | 72.95        | 52.86          | 41.56                  | 76.11         | 37.07                  | 66.91                  | 38.73                      | 80.66                   | 56.58         | 56.99 ± 1.11            |

### Device-wise results

|   **Split** | **A**   | **B**   | **C**   | **S1**   | **S2**   | **S3**   | **S4**   | **S5**   | **S6**   | **Macro Average Accuracy**  |
|---------:|:-------------|:-------------|:-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:---------------------|
|        **5%** | 54.45      | 45.73      | 48.42      | 39.66       | 36.13       | 44.30       | 38.90       | 40.47       | 33.58       | 42.40 ± 0.42              |
|       **10%** | 57.84      | 48.60      | 51.13      | 42.16       | 40.30       | 46.00       | 43.13       | 41.30       | 37.26       | 45.29 ± 1.01              |
|       **25%** | 62.27      | 53.27      | 55.39      | 47.52       | 46.68       | 51.59       | 47.39       | 46.75       | 41.75       | 50.29 ± 0.87               |
|       **50%** | 65.39      | 56.30      | 57.23      | 52.99       | 50.85       | 54.78       | 48.35       | 47.93       | 44.90       | 53.19 ± 0.68              |
|      **100%** | 67.17      | 59.67      | 61.99      | 56.28       | 55.69       | 58.16       | 53.05       | 52.35       | 48.58       | 56.99 ± 1.11              |

## Obtain Evaluation set predictions

The evaluation set will be published on **1st of June**. Details on the submission guidelines can be found in the [official task description](https://dcase.community/challenge2024/).

The evaluation set comes without corresponding labels and you will submit csv files containing the evaluation set predictions. We provide
an example on how to generate predictions on the unlabeled evaluation set. 

1. Download and extract the evaluation set. You will obtain a folder *TAU-urban-acoustic-scenes-2024-mobile-evaluation* containing, among other files, a folder *audio* and a file *meta.csv*. It must be located in the same directory
as *TAU-urban-acoustic-scenes-2022-mobile-development*.
2. Generate evaluation set predictions for a trained system by running the command:

```python run_training.py --evaluate --ckpt_id=<wandb_id>```

This command will load a checkpoint automatically created by a previous experiment. Checkpoints are stored in a folder named *DCASE24_Task1*. You can select the experiment by 
specifying its ID (assigned by weights and biases). 

The script will generate a folder *predictions/<wandb_id>* containing the predictions on the evaluation set (*output.csv*),
the model's performance on the development-test split (*info.json*), and the model's state dict.


