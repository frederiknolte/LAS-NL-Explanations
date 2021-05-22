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

This section describes how to run the LAS model on TPUs.

### Setting up the Environment

1. Connect to your TPU

2. Download the Miniconda installer:
   ```shell
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
    ```

3. Install Anaconda:
   ```shell
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
   
4. Create the Environment:
   ```shell
    conda create -n cdm python=3.7
    ```
    Now log out and log back in again.
   
5. Activate the Environment:
   ```shell
   source activate cdm
    ```

6. Install the required packages:
   ```shell
    pip install -r LAS/requirements.txt
    ```

### Starting the Training on TPU

1. Go to the Google Cloud Console on your browser and visit the TPU section. Copy the IP address of the TPU instance.

2. Execute the following in the TPU shell:
   ```shell
   export TPU_IP_ADDRESS=your-TPU-IP-address; \
   export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    ```

3. Move to the correct directory:
   ```shell
   cd LAS-NL-Explanations/sim_experiments/
    ```

4. Start the training:
   ```shell
   python run_tasks.py -e circa.NLI.SIM.ST.RE -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir --server_number 1 --model t5-small --use_tpu
    ```
