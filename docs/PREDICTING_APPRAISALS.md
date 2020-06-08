# Predicting Appraisals from Text
- [General Information](#general-information)
  * [Embedding Setup](#embedding-setup)
- [Example Usage](#example-usage)
- [Creating Annotations using Predicted Appraisals](#creating-annotations-using-predicted-appraisals)
- [Tips](#tips)
- [Help](#help)

## General Information
This repository provides a Convolutional Neural Network for predicting appraisal
dimensions from text.

### Embedding Setup
If you are using your own dataset it is important to setup a word embedding.
This can be done using the `--createembedding (-ce)` argument. You also need to
download a 300 dimensional [GloVe](https://nlp.stanford.edu/projects/glove/)
embedding and specify the path to this download using the `--embeddingpath (-ep)`
argument. Usage:
```bash
python3 b_appraisals_from_text.py --dataset <AppraisalDataset> --createembedding <EmbeddingSaveFile.npy> --embeddingpath <path/to/glove300.txt>
```

## Example Usage
* Performing cross validation on your dataset:
```bash
python3 b_appraisals_from_text.py --dataset <AppraisalDataset> --loadembedding <EmbeddingSaveFile.npy>
```

* Using a train and test set:
```bash
python3 b_appraisals_from_text.py --dataset <AppraisalTrainSet> --testset <AppraisalTestSet> --loadembedding <EmbeddingSaveFile.npy>
```

* Saving appraisal prediction model for later use:
```bash
python3 b_appraisals_from_text.py --dataset <AppraisalTrainSet> --savemodel <SaveFile.h5>
```

* Loading saved appraisal prediction model:
```bash
python3 b_appraisals_from_text.py --dataset <AppraisalTestSet> --loadmodel <SaveFile.h5>
```

## Creating Annotations using Predicted Appraisals
This tool can be used for creating appraisal annotations on a dataset
with no emotion labels. To do this

## Tips
* Specify the number of folds using the `--folds/-f` argument
* Specify the number of CV runs using the `--runs/-r` argument
* Specify the number of training epochs using the `--epochs/-e` argument
* Specify the training batchsize using the `--batchsize/-b` argument



* Change the final evaluation report format to *latex* or *markdown* using the `--format`argument
* Use the `--help/-h` argument for additional help

## Help
```
usage: b_appraisals_from_text.py [-h] --dataset DATASET [--testset TESTSET]
                                 [--annotate ANNOTATE] [--savemodel SAVEMODEL]
                                 [--loadmodel LOADMODEL]
                                 [--loadembedding LOADEMBEDDING]
                                 [--createembedding CREATEEMBEDDING]
                                 [--embeddingpath EMBEDDINGPATH]
                                 [--epochs EPOCHS] [--batchsize BATCHSIZE]
                                 [--folds FOLDS] [--runs RUNS] [--continous]
                                 [--format {text,latex,markdown}] [--gpu]
                                 [--cpu] [--quiet] [--debug]

required arguments:
  --dataset DATASET, -d DATASET
                        specify the input dataset.
                        Corresponds to the training set if you also specify a testset

optional arguments:
  --testset TESTSET, -t TESTSET
                        if you have a test set and you don't want to use cross-validation
                        you can specify a test set here.
                         Then the --dataset argument must be your training set
  --annotate ANNOTATE, -a ANNOTATE
                        annotate a dataset using the predicted appraisals
  --savemodel SAVEMODEL, -s SAVEMODEL
                        if you do not want to run a cross-validation you can save
                        the created appraisal prediction model weights and use them in other experiments
  --loadmodel LOADMODEL, -l LOADMODEL
                        test your saved models
                        The dataset specified with the --dataset command will be your test set

embedding configuration:
  --loadembedding LOADEMBEDDING, -le LOADEMBEDDING
                        path to your embedding file created for your dataset
  --createembedding CREATEEMBEDDING, -ce CREATEEMBEDDING
                        path to the embedding file for your dataset, which will be created
                        with all words in your dataset
  --embeddingpath EMBEDDINGPATH, -ep EMBEDDINGPATH
                        path to your pre-trained embedding download (e.g. glove300)
                        note that only 300 dimensional embeddings are supported

parameter configuration:
  --epochs EPOCHS, -e EPOCHS
                        set the number training epochs (default: 20)
  --batchsize BATCHSIZE, -b BATCHSIZE
                        set the training batchsize (default: 32)
  --folds FOLDS, -f FOLDS
                        set the number of CV folds (default: 10)
  --runs RUNS, -r RUNS  set the number of CV runs (default: 10)
  --continous, -c       use this if you are using continous-valued appraisal annotation
                        instead of binary

optional tensorflow setup:
  --format {text,latex,markdown}
                        final result output format (default: text)
  --gpu                 force to run experiment on GPU
  --cpu                 force to run experiment on CPU
  --quiet               reduce keras and info outputs
  --debug               show tensorflow backend informations
```
