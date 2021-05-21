# Running the LAS Model

## Preparing the Data
Data preparation is handled by the ``reformat_circa.py`` file. First, you need to download the input, target and prediction file from the Google 
Cloud Bucket, which can easily be done through the browser interface. After, moving the three files into this project directory, you call 
``reformat_circa.py`` with specifying the location of the three files. The resulting train/dev/test splits will be saved in 
``sim_experiments/data/circa/{QA/NLI}``.

## Running LAS on LISA
This section describes how to run the LAS model on LISA. [Running experiments on TPUs](Running-LAS-on-TPU) is described in a later section.

### Setting up the environment
1. Load the required modules by executing:
    ```shell
    module purge
    module load 2019
    module load Python/3.7.5-foss-2019b
    module load cuDNN/7.6.5.32-CUDA-10.1.243
    module load NCCL/2.5.6-CUDA-10.1.243
    module load Anaconda3/2018.12
    ```

2. Then, create a new conda environment:
    ```shell
    conda create -n cdm python=3.7
    ```

3. Now activate it and install the necessary packages:
    ```shell
    source activate cdm
    pip install -r LAS/requirements.txt
    ```

Now log out of LISA and log back in again.

### Training the Simulator

1. Move to the correct directory:
    ```shell
    cd LAS-NL-Explanations/sim_experiments/
    ```

2. Make sure that the following directory exists:
    ```shell
    mkdir outputs
    ```

3. The training will be started by executing:
    ```shell
    sbatch train_circa.job
    ```

The job file contains several settings:
- ``-e`` marks the difference between an NLI and QA task. For NLI, set `circa.NLI.SIM.ST.RE`. For QA, set `circa.QA.SIM.MT.RE`.
- ``-b`` denotes the batch size.
- ``-g`` denotes the number of gradient accumulations. The effective batch size is the product of ``-b``and ``-g``.

Note that you might want to adjust the maximal job time of the Slurm scheduler.

### Evaluating the LAS Score
The following can only be executed if the [simulator is trained](Training-the-Simulator).

1. Move to the correct directory:
    ```shell
    cd LAS-NL-Explanations/sim_experiments/
    ```

2. The LAS scoring will be started by executing:
    ```shell
    sbatch LAS_circa.job
    ```

The job file contains the following setting:
- ``--data`` Select `circa_NLI` for NLI tasks and `circa_QA` for QA tasks.

## Running LAS on TPU
TBD
