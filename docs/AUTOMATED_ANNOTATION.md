# Automated Appraisal Annotation
- [General Information](#general-information)
- [Usage](#usage)
- [Annotation Variants](#annotation-variants)
- [Help](#help)

## General Information
You can create automated appraisal annotations on other datasets using the tool
`x_create_automated_annotation.py`.
This means you can experiment with all 15 class labeles provided by [Smith and Ellsworth (1985)](https://www.researchgate.net/publication/19274815_Patterns_of_Cognitive_Appraisal_in_Emotion, 'https://www.researchgate.net/publication/19274815_Patterns_of_Cognitive_Appraisal_in_Emotion').

Supported class labels are:  
```
Anger, Boredom, Challenge, Contempt, Disgust, Fear, Frustration,
Guilt, Hope, Interest, Joy, Pride, Sadness, Shame, Surprise
```

Note that it is important, that class labels in your dataset are exactly named as mentioned above.
In addition, the dataset must have the following format:
```
<ClassLabel>,<TextInstance>
or
<CLassLabel>[TAB]<TextInstance>

Examples:
Joy,When I was walking outside on a sunny day.
or
Joy[TAB]When I was walking outside on a sunny day.
```

The tool will create a new dataset with appraisal annotations in the following form:
```bash
Prior_Emotion	Sentence	Attention	Certainty	Effort	Pleasant	Responsibility/Control	Situational Control
```


## Usage
* Creating a dataset annotated with appraisals:
```bash
python3 x_create_automated_annotation.py --dataset <InputDataset>
```

* The tool will automatically find all labels your dataset contains. However,
you can also specify the labels using arguments:
```bash
python3 x_create_automated_annotation.py --dataset <InputDataset> --labels Anger Joy Sadness
```

* If you want to annotate only a subset of the Appraisals you can use
--appraisals argument. The following example will create an annotation only containing
the appraisals Attention Certainty and Pleasantness.
```bash
python3 x_create_automated_annotation.py --dataset <InputDataset> --appraisals A C P
```

## Annotation Variants
There are possible ways of annotating:

* Binary-valued appraisal annotation using 'zero threshold' normalization
* Binary-valued appraisal annotation values using Min-Max normalization
* Continous valued annotation (Use the original appraisal  values)

You can specify them using arguments
```bash
python3 x_create_automated_annotation.py --dataset <InputDataset> -n zero
python3 x_create_automated_annotation.py --dataset <InputDataset> -n minmax
python3 x_create_automated_annotation.py --dataset <InputDataset> -n none
```

## Help
```
usage: x_create_automated_annotation.py [-h] --dataset DATASET
                                        [--output OUTPUT]
                                        [--normalization {zero,minmax,none}]
                                        [--labels LABELS [LABELS ...]]
                                        [--appraisals {A,C,E,P,RC,S} [{A,C,E,P,RC,S} ...]]

required arguments:
  --dataset DATASET, -d DATASET
                        specify an input dataset. supported files are .csv and .tsv
                        note that instances in the dataset must have the following formatting:
                        Class (Emotion) Label <TAB> Text
                        	 or
                        Class (Emotion) Label , Text

                        This means instances should look like this:
                        Joy,When I was outside on a sunny day.

optional arguments:
  --output OUTPUT, -o OUTPUT
                        specify an output name for the automated appraisal annotation dataset
  --normalization {zero,minmax,none}, -n {zero,minmax,none}
                        choose normaization method. Options:
                        zero   - possitive values will be annotated with 1 and negative values with 0
                        minmax - experimental min-max scaling
                        none   - use original continous values
  --labels LABELS [LABELS ...], -l LABELS [LABELS ...]
                        set labels contained in the dataset
                        supported labels are:
                        Anger, Boredom, Challenge, Contempt, Disgust
                        Fear, Frustration, Guilt, Hope, Interest
                        Joy, Pride, Sadness, Shame, Surprise
  --appraisals {A,C,E,P,RC,S} [{A,C,E,P,RC,S} ...], -a {A,C,E,P,RC,S} [{A,C,E,P,RC,S} ...]
                        specify appraisals you want to annotate
                        (default: Attention, Certainty, Effort, Pleasantness, Responsibility/Control, Situational Control)

                        you have to use abbreviations:
                        Attention: A, Certainty: C, Effort: E, Pleasantness: P
                        Responsibility/Control: RC, Situational Control: S, i.e.
                        --appraisals A C E
                        if you only want to use "Attention", "Certainty" and "Effort"
```
