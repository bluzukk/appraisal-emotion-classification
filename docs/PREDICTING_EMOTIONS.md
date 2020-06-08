# Predicting Emotions from Appraisals
- [Example Usage](#example-usage)
- [Tips](#tips)
- [Help](#help)

## Example Usage
* Performing cross validation on your dataset:
```bash
python3 a_emotions_from_appraisals.py --dataset <AppraisalDataset>
```

* Using a train and test set:
```bash
python3 a_emotions_from_appraisals.py --dataset <AppraisalTrainSet> --testset <AppraisalTestSet>
```

* Saving emotion prediction model for later use:
```bash
python3 a_emotions_from_appraisals.py --dataset <AppraisalTrainSet> --savemodel <SaveFile.h5>
```

* Loading saved appraisal prediction model:
```bash
python3 a_emotions_from_appraisals.py --dataset <AppraisalTestSet> --loadmodel <SaveFile.h5>
```

## Tips
* Specify the number of folds using the `--folds/-f` argument
* Specify the number of CV runs using the `--runs/-r` argument
* Specify the number of training epochs using the `--epochs/-e` argument
* Specify the training batchsize using the `--batchsize/-b` argument

* Use the `--help/-h` argument for additional help

## Help
```
usage: a_emotions_from_appraisals.py [-h] --dataset DATASET
                                     [--testset TESTSET]
                                     [--savemodel SAVEMODEL] [--epochs EPOCHS]
                                     [--batchsize BATCHSIZE] [--folds FOLDS]
                                     [--runs RUNS] [--gpu] [--cpu] [--quiet]
                                     [--debug]

required arguments:
  --dataset DATASET, -d DATASET
                        specify the input dataset.
                        Corresponds to the training set if you also specify a testset

optional arguments:
  --testset TESTSET, -t TESTSET
                        if you have a test set and you don't want to use cross-validation
                        you can specify a test set here.
                         Then the --dataset argument must be your training set
  --savemodel SAVEMODEL, -s SAVEMODEL
                        If you do not want to run a cross-validation you can save
                        the created emotion prediction model weights and use them in other experiments

parameter configuration:
  --epochs EPOCHS, -e EPOCHS
                        set the number training epochs (default: 20)
  --batchsize BATCHSIZE, -b BATCHSIZE
                        set the training batchsize (default: 32)
  --folds FOLDS, -f FOLDS
                        set the number of CV folds (default: 10)
  --runs RUNS, -r RUNS  set the number of CV runs (default: 10)

optional tensorflow setup:
  --gpu                 force to run experiment on GPU
  --cpu                 force to run experiment on CPU
  --quiet               reduce keras outputs
  --debug               show tensorflow backend informations
```
