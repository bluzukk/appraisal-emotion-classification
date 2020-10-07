This repository contains the Appraisal enISEAR dataset: A reannotation of the enISEAR corpus with Cognitive Appraisal.
Further, this repository contains python code that was used for the experiments in the paper 
*[Appraisal Theories for Emotion Classification in Text](https://arxiv.org/abs/2003.14155).*  



*Jan Hofmann, Enrica Troiano, Kai Sassenberg, and Roman Klinger. 
Appraisal theories for emotion classification in text. 
In Proceedings of the 28th International Conference on Computational Linguistics, 2020. 
[ [bib](https://www.romanklinger.de/publications/2020_bib.html#Hofmann2020b) | [http](https://arxiv.org/abs/2003.14155)]*


In addition, this repository provides scripts to reproduce the results of the models reported in the paper. 
For more information about the Appraisal enISEAR dataset see the [Corpora Overview](corpora).

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
- [Annotating unlabeled Text Instances with Appraisals](#annotating-unlabeled-text-instances-with-appraisals)

# Requirements

## Installation using Anaconda
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Install Tensorflow environment

  ```
  conda create -n tf tensorflow=1.14  # (CPU-Version)
  ```

  ```
  conda create -n tf tensorflow-gpu=1.14 # (GPU-Version)
  ```

3. Activate Tensorflow environment and install futher dependencies

  ```
  conda activate tf;
  pip3 install numpy==1.16.4 pandas sklearn keras==2.3.0 gensim;
  ```

## Manual Installation

#### CPU Only
* Python 3.5+
* Install dependencies:
```
pip3 install --user numpy==1.16.4 pandas sklearn keras==2.3.0 gensim tensorflow==1.14
```

#### GPU Support (nvidia)
* Python 3.5 or 3.6
* Install dependencies:
```
pip3 install --user numpy==1.16.4 pandas sklearn keras==2.3.0 gensim tensorflow-gpu==1.14
```

* Install [cuda](https://developer.nvidia.com/cuda-downloads 'https://developer.nvidia.com/cuda-downloads') (Testet with cuda 10.0.130)
* Install [cuDNN](https://developer.nvidia.com/cudnn 'https://developer.nvidia.com/cudnn') (Tested with cudnn 7.6.5.32-2)

This configuration was tested on linux with python 3.8.1 for CPU and python 3.5.9 for GPU support.


# Reproducing Results on 'Appraisal enISEAR'

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

# Annotating unlabeled Text Instances with Appraisals
The scripts can be used to predict appraisals on text instances with no emotions labels.  
To do this first navigate to the ''main experiment'' directory with `cd impl-main-experiments;`  
Then run the `b_appraisals_from_text.py` script with the `--annotate` option:

```bash
python3 b_appraisals_from_text.py -d enISEAR_V1 --annotate <TextInstances>.csv
```

<br>

You can also skip training to predict appraisals by using the pre-trained model, which was
trained to predict appraisals on 'appraisal enISEAR'  
To annotate your text instances using the pre-trained model run:

```bash
python3 b_appraisals_from_text.py -d enISEAR_V1 --annotate <TextInstances>.csv --loadmodel ../pre-trained/enISEAR_V1_appraisal_predictor.h5
```

<br>
Your dataset annotated with predicted appraisals will be saved to 'TextInstances_appraisals.csv'  
<br>
Note that your dataset musst be formatted as follows:

```
Sentence
<Text Instance 1>
<Text Instance 2>
<Text Instance 3>
.
.
.
```
This means the first line in the file musst be 'Sentence' followed by your text instances (one in every line).
