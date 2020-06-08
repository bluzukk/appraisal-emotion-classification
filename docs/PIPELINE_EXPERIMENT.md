# Text Appraisal Emotion Pipeline Experiment
- [General Information](#general-information)
  * [Embedding Setup](#embedding-setup)
- [Example Usage](#example-usage)
- [Tips](#tips)
- [Help](#help)

## General Information

### Embedding Setup
If you are using your own dataset it is important to setup a word embedding.
This can be done using the `--createembedding (-ce)` argument. You also need to
download a 300 dimensional [GloVe](https://nlp.stanford.edu/projects/glove/)
embedding and specify the path to this download using the `--embeddingpath (-ep)`
argument. Usage:
```bash
python3 c_pipeline.py --dataset <AppraisalDataset> --createembedding <EmbeddingSaveFile.npy> --embeddingpath <path/to/glove300.txt>
```

## Example Usage
* Performing cross validation on your dataset:
```bash
python3 c_pipeline.py --dataset <AppraisalDataset> --loadembedding <EmbeddingSaveFile.npy>
```

* Using a train and test set:
```bash
python3 c_pipeline.py --dataset <AppraisalTrainSet> --testset <AppraisalTestSet> --loadembedding <EmbeddingSaveFile.npy>
```

* Saving pipeline model for later use:
```bash
python3 c_pipeline.py --dataset ISEAR.tsv --savemodel <ApraisalModel.h5> <EmotionModel.h5> --loadembedding <EmbeddingSaveFile.npy>
```

* Loading saved pipeline models:
```bash
python3 c_pipeline.py --dataset ISEAR.tsv --loadmodel <ApraisalPredictionModel.h5> <EmotionPredictionModel.h5> --loadembedding <EmbeddingSaveFile.npy>
```

* Learning 'text-appraisal-emotion' pipeline on using *enISEAR* appraisal and emotion annotations and testing the
model on your dataset:
```bash
python3 c_pipeline.py --dataset enISEAR_V1 --testset <YourTestSet>
```

## Tips
* Specify the number of folds using the `--folds/-f` argument
* Specify the number of CV runs using the `--runs/-r` argument
* Specify the number of training epochs using the `--epochs/-e` argument
* Specify the training batchsize using the `--batchsize/-b` argument

* Use the `--help/-h` argument for additional help


## Help
```
usage: c_pipeline.py [-h] --dataset DATASET [--testset TESTSET]
                     [--savemodel SAVEMODEL [SAVEMODEL ...]]
                     [--loadmodel LOADMODEL [LOADMODEL ...]]
                     [--loadembedding LOADEMBEDDING]
                     [--createembedding CREATEEMBEDDING]
                     [--embeddingpath EMBEDDINGPATH]
                     [--epochs_appraisal EPOCHS_APPRAISAL]
                     [--epochs_emotion EPOCHS_EMOTION] [--batchsize BATCHSIZE]
                     [--folds FOLDS] [--runs RUNS] [--continous] [--gpu]
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
  --savemodel SAVEMODEL [SAVEMODEL ...], -s SAVEMODEL [SAVEMODEL ...]
                        If you do not want to run a cross-validation you can save
                        the created emotion prediction model weights and use them in other experiments
                        usage: --savemodel <APPRAISAL_MODEL.h5> <EMOTION_MODEL.h5>
  --loadmodel LOADMODEL [LOADMODEL ...], -l LOADMODEL [LOADMODEL ...]
                        test your saved models
                        The dataset specified with the --dataset command will be your test set
                        usage: --loadmodel <APPRAISAL_MODEL.h5> <EMOTION_MODEL.h5>

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
  --epochs_appraisal EPOCHS_APPRAISAL, -ea EPOCHS_APPRAISAL
                        set the number training epochs (default: 20)
  --epochs_emotion EPOCHS_EMOTION, -ee EPOCHS_EMOTION
                        set the number training epochs (default: 20)
  --batchsize BATCHSIZE, -b BATCHSIZE
                        set the training batchsize (default: 32)
  --folds FOLDS, -f FOLDS
                        set the number of CV folds (default: 10)
  --runs RUNS, -r RUNS  set the number of CV runs (default: 10)
  --continous, -c       use this if you are using continous-valued appraisal annotation
                        instead of binary

optional tensorflow setup:
  --gpu                 force to run experiment on GPU
  --cpu                 force to run experiment on CPU
  --quiet               reduce keras and info outputs
  --debug               show tensorflow backend informations
```
