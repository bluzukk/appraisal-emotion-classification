*This repository contains the Appraisal enISEAR dataset. A reannotation of the enISEAR corpus with Cognitive Appraisal.*  
*Further, this repository contains python code that was used for the experiments in the paper*  
*[Appraisal Theories for Emotion Classification in Text](https://arxiv.org/abs/2003.14155).*  

*In addition, this repository provides scripts to reproduce the results of the models reported in the paper.*  
*For more information about the Appraisal enISEAR dataset see the [Corpora Overview](corpora/README.md).*

<br>
<br>

- [Requirements](#requirements)
  * [Installation using Anaconda](#installation-using-anaconda)
  * [Manual Installation](#manual-installation)
- [Reproducing Results on enISEAR](#reproducing-results-on-enisear)
  * [Baseline Results](#baseline-results)
  * [Appraisal predictions based on Text](#appraisal-predictions-based-on-text)
  * [Emotion predictions based on Appraisals](#emotion-predictions-based-on-appraisals)
  * [Pipeline Experiment](#pipeline-experiment)
  * [Multi-task and Oracle Experiment](#multi-task-and-oracle-experiment)
- [Experimenting with Appraisal Models](#experimenting-with-appraisal-models)
  * [Creating automated Appraisal Annotations](#creating-automated-appraisal-annotations)
  * [Experimenting with automated Appraisal Annotations](#experimenting-with-automated-appraisal-annotations)
    + [Predicting Emotions from Appraisals](#predicting-emotions-from-appraisals)
    + [Predicting Appraisals from Text](#predicting-appraisals-from-text)
    + [Pipeline Experiment](#pipeline-experiment-1)

# Requirements

## Installation using Anaconda
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Install Tensorflow environment

  ```
  conda create -n tf tensorflow  # (CPU-Version)
  ```

  ```
  conda create -n tf tensorflow-gpu # (GPU-Version)
  ```

3. Activate Tensorflow environment and install futher dependencies

  ```
  conda activate tf;
  pip3 install numpy==1.16.4 pandas sklearn keras gensim;
  ```

## Manual Installation

#### CPU Only
* Python 3.5+
* Install dependencies:
```
pip3 install --user numpy==1.16.4 pandas sklearn keras gensim tensorflow
```

#### GPU Support (nvidia)
* Python 3.5 or 3.6
* Install dependencies:
```
pip3 install --user numpy==1.16.4 pandas sklearn keras gensim tensorflow-gpu==1.14
```

* Install [cuda](https://developer.nvidia.com/cuda-downloads 'https://developer.nvidia.com/cuda-downloads') (Testet with cuda 10.0.130)
* Install [cuDNN](https://developer.nvidia.com/cudnn 'https://developer.nvidia.com/cudnn') (Tested with cudnn 7.6.5.32-2)

This configuration was tested on linux with python 3.8.1 for CPU and python 3.5.9 for GPU support.


# Reproducing Results on enISEAR

## Baseline Results
To start the baseline models on the enISEAR dataset navigate to the baseline directory with `cd impl-z-baseline;` then run
```bash
python3 baseline_CNN.py --dataset enISEAR
```
```bash
python3 baseline_MaxEnt.py --dataset enISEAR
```

## Appraisal predictions based on Text
In order to reproduce the ''appraisal prediction from text'' experiment navigate to the ''main experiment'' directory with `cd impl-main-experiments;`. Then run
```bash
python3 a_appraisals_from_text.py --dataset enISEAR_V1
```

To run the model on the *automated annotations* run
```bash
python3 a_appraisals_from_text.py --dataset enISEAR_V3
```

## Emotion predictions based on Appraisals
In order to predict emotions from appraisals navigate to the ''main experiment'' directory with `cd impl-main-experiments;`. Then run
```bash
python3 b_emotions_from_appraisals.py --dataset enISEAR_V1
```

## Pipeline Experiment
In order to reproduce the ''appraisal-emotion-pipeline'' experiment navigate to the ''main experiment'' directory with `cd impl-main-experiments;`. Then run
```bash
python3 c_pipeline.py --dataset enISEAR_V1
```

## Multi-task and Oracle Experiment
In order to reproduce the multi-task experiment navigate to the additional experiments directory with `cd impl-x-additional-experiments;`. Then run
```bash
python3 multi_task_fully_shared.py --dataset enISEAR_V1
```
or
```bash
python3 oracle_CNN_saveModels.py --dataset enISEAR_V1 --rounds 1
```
