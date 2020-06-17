# Corpora

This repository contains the *Appraisal enISEAR* dataset. A reannotation of the
enISEAR corpus with Cognitive Appraisal.

If you are working with *Appraisal enISEAR* please consider citing the paper on the original enISEAR dataset:  
[Crowdsourcing and Validating Event-focused Emotion Corpora for German and English (2019)](https://www.aclweb.org/anthology/P19-1391/).

<br>
<br>

There are three versions of the *Appraisal enISEAR*:
* enISEAR-appraisal-V1: A binary-valued manual annotation. (This is the annotation used in the paper)
* enISEAR-appraisal-V2: A continous-valued automatic annotation based on emotion labels.
* enISEAR-appraisal-V3: A binary-valued automatic annotation based on emotion labels.


<br>
<br>

#### Directory Overview
``` bash
├── enISEAR-appraisal-V1
│   ├── enISEAR_appraisal_a1.tsv
│   ├── enISEAR_appraisal_a2.tsv
│   ├── enISEAR_appraisal_a3.tsv
│   ├── enISEAR_appraisal_high_recall.tsv
│   ├── enISEAR_appraisal_majority.tsv
│   ├── enISEAR_appraisal_merged.tsv
│   └── enISEAR_appraisal.tsv
├── enISEAR-appraisal-V2
│   └── enISEAR_appraisals_automated_continous.tsv
└── enISEAR-appraisal-V3
    └── enISEAR_appraisal_automated_binary.tsv

```
