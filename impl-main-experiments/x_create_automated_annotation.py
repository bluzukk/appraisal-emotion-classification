import csv
import argparse
from argparse import RawTextHelpFormatter
import sys

"""

Tool for creating an automated annotation based on appraisal dimensions.

    - automatically assign appraisal annotations to instances labeled with emotions.
    - supported labels: Anger, Boredom, Challenge, Contempt, Disgust
                        Fear, Frustration, Guilt, Hope, Interest
                        Joy, Pride, Sadness, Shame, Surprise

--------------------------------------------------------
# Original values derived by [Smith85]
              Pleas. Re.Co.  Cert.  Att.   Eff.   Sit.
Happiness     -1.46   0.09  -0.46   0.15  -0.33  -0.21
Sadness        0.87  -0.36   0.00  -0.21  -0.14   1.15
Anger          0.85  -0.94  -0.29   0.12   0.53  -0.96
Boredom        0.34  -0.19  -0.35  -1.27  -1.19   0.12
Challenge     -0.37   0.44  -0.01   0.52   1.19  -0.20
Hope          -0.50   0.15   0.46   0.31  -0.18   0.35
Fear           0.44  -0.17   0.73   0.03   0.63   0.59
Interest      -1.05  -0.13  -0.07   0.70  -0.07  -0.63
Contempt       0.89  -0.50  -0.12   0.08  -0.07  -0.63
Disgust        0.38  -0.50  -0.39  -0.96   0.06  -0.19
Frustration    0.88  -0.37  -0.08   0.60   0.48   0.22
Surprise      -1.35  -0.94   0.73   0.40  -0.66   0.15
Pride         -1.25   0.81  -0.32   0.02  -0.31  -0.46
Shame          0.73   1.31   0.21  -0.11   0.07  -0.07
Guilt          0.60   1.31  -0.15  -0.36   0.00  -0.29
--------------------------------------------------------

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
"""

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--dataset', '-d',
            type=str,
            help='specify an input dataset. supported files are .csv and .tsv\n'
                 'note that instances in the dataset must have the following formatting:\n'
                 'Class (Emotion) Label <TAB> Text\n'
                 '\t or \n'
                 'Class (Emotion) Label , Text\n'
                 '\n'
                 'This means instances should look like this:\n'
                 'Joy,When I was outside on a sunny day.',
            required=True)
optional.add_argument('--output', '-o',
            type=str,
            help='specify an output name for the automated appraisal annotation dataset',
            required=False)
optional.add_argument('--normalization', '-n',
            default='zero', type=str,
            choices=['zero', 'minmax', 'none'],
            help='choose normaization method. Options:\n'
                 'zero   - possitive values will be annotated with 1 and negative values with 0\n'
                 'minmax - experimental min-max scaling\n'
                 'none   - use original continous values\n')
optional.add_argument('--labels', '-l',
            nargs='+',
            help='set labels contained in the dataset\n'
                 'supported labels are:\n'
                 'Anger, Boredom, Challenge, Contempt, Disgust\n'
                 'Fear, Frustration, Guilt, Hope, Interest\n'
                 'Joy, Pride, Sadness, Shame, Surprise')
optional.add_argument('--appraisals', '-a',
            nargs='+',
            choices=['A', 'C', 'E', 'P', 'RC', 'S'],
            help='specify appraisals you want to annotate \n'
            '(default: Attention, Certainty, Effort, Pleasantness, Responsibility/Control, Situational Control) \n\n'
            'you have to use abbreviations:\n'
            'Attention: A, Certainty: C, Effort: E, Pleasantness: P\n'
            'Responsibility/Control: RC, Situational Control: S, i.e.\n'
            '--appraisals A C E\n'
            'if you only want to use "Attention", "Certainty" and "Effort"\n',
            required=False)
args = parser.parse_args()


# If no output file specified use default output file
input_file = args.dataset
if (not args.output):
    if (input_file.endswith('.csv') or input_file.endswith('.tsv')):
        filename = input_file[:-4]
        output_file = filename + '_appraisals.tsv'
else: output_file = args.output

# Load labels and text instances from the input dataset
text_instance  = []
emotion_labels = []
firstline = True # Skip first line
try:
    with open(input_file) as csvfile:
        if (input_file.endswith('.csv')):
            csvreader = csv.reader(csvfile, delimiter=',')
        else:
            csvreader = csv.reader(csvfile, delimiter='\t')
        for line in csvreader:
            if not firstline:
                emotion_labels.append(line[0].capitalize())
                text_instance.append(line[1])
            firstline = False
except FileNotFoundError:
    print('\nERROR: File %s was not found' % input_file)
    print('\nExiting.')
    sys.exit(1)
except:
    print('\nERROR: Error reading input file %s' % input_file)
    print('       Make sure your dataset is correctly formatted and')
    print('       Also has the correct file extension.')
    print('       use .csv for comma separated values')
    print('       use .tsv for tab separated values')
    print('')
    print(sys.exc_info()[1])
    sys.exit(1)

# If no labels specified use following default labels
if (not args.labels):
    LABELS = sorted(list(set(emotion_labels)))
else: LABELS = args.labels

APPRAISALS_ALL = ['Attention', 'Certainty', 'Effort', 'Pleasantness', 'Responsibility/Control', 'Situational Control']
if (not args.appraisals):
    APPRAISALS = APPRAISALS_ALL
else:
    APPRAISALS = []
    if ('A' in args.appraisals):
        APPRAISALS.append('Attention')
    if ('C' in args.appraisals):
        APPRAISALS.append('Certainty')
    if ('E' in args.appraisals):
        APPRAISALS.append('Effort')
    if ('P' in args.appraisals):
        APPRAISALS.append('Pleasantness')
    if ('RC' in args.appraisals):
        APPRAISALS.append('Responsibility/Control')
    if ('S' in args.appraisals):
        APPRAISALS.append('Situational Control')

print('''\n---------------------------------------''')
print('''  Automated Appraisal annotation Tool''')
print('''---------------------------------------''')
print('  Configuration:')
print('    Input File    : %s' % input_file)
print('    Output File   : %s' % output_file)
print('    Class Labels  : %s' % LABELS)
print('    Appraisals    : %s' % APPRAISALS)
print('    Normalization : %s' % args.normalization)
print('''---------------------------------------''')

def normalize_zero_threshold(data):
    normalized = []
    for row in data:
        norm_row = []
        for value in row:
            if (value >= 0):
                value = 1
            else: value = 0
            norm_row.append(value)
        normalized.append(norm_row)
    return normalized

def normalize_min_max(data):
    min = -0.94
    max =  1.46
    normalized = []
    for row in data:
        norm_row = []
        for value in row:
            if ((value-min)/(max-min) > 0.5):
                value = 1
            else: value = 0
            norm_row.append(value)
        normalized.append(norm_row)
    return normalized

# Rearrange appraisal data
# Note that Certainty and Pleasantness values are 'inverted'
appraisal_ratings = {"Joy":         [0.15,   0.46, -0.33,  1.46,  0.09, -0.21],
                     "Sadness":     [-0.21,  0.00, -0.14, -0.87, -0.36,  1.15],
                     "Anger":       [0.12,   0.29,  0.53, -0.85, -0.94, -0.96],
                     "Boredom":     [-1.27,  0.35, -1.19, -0.34, -0.19,  0.12],
                     "Challenge":   [0.52,   0.01,  1.19,  0.37,  0.44, -0.20],
                     "Hope":        [0.31,  -0.46, -0.18,  0.50,  0.15,  0.35],
                     "Fear":        [0.03,  -0.73,  0.63, -0.44, -0.17,  0.59],
                     "Interest":    [0.70,   0.07, -0.07,  1.05, -0.13, -0.63],
                     "Contempt":    [0.08,   0.12, -0.07, -0.89, -0.50, -0.63],
                     "Disgust":     [-0.96,  0.39,  0.06, -0.38, -0.50, -0.19],
                     "Frustration": [0.60,   0.08,  0.48, -0.88, -0.37,  0.22],
                     "Surprise":    [0.40,  -0.73, -0.66,  1.35, -0.94,  0.15],
                     "Pride":       [0.02,   0.32, -0.31,  1.25,  0.81, -0.46],
                     "Shame":       [-0.11, -0.21,  0.07, -0.73,  1.31, -0.07],
                     "Guilt":       [-0.36,  0.15,  0.00, -0.60,  1.31, -0.29]}

_appraisal_ratings = []
for label in LABELS:
    try:
        _appraisal_ratings.append(appraisal_ratings[label])
    except KeyError:
        print('\nERROR: Label "%s" is not valid' % label)
        print('Make sure your class labels are a subset of the following:')
        print('\tAnger, Boredom, Challenge, Contempt, Disgust')
        print('\tFear, Frustration, Guilt, Hope, Interest')
        print('\tJoy, Pride, Sadness, Shame, Surprise')
        print('\nExiting')
        sys.exit(1)

if (args.normalization == 'zero'):
    # Use 'zero threshold' binarization
    normalized_data = normalize_zero_threshold(_appraisal_ratings)
elif (args.normalization == 'minmax'):
    # Apply minmax scaling
    normalized_data = normalize_min_max(_appraisal_ratings)
    # Use original values
else: normalized_data = _appraisal_ratings

# if (not args.normalization == 'none'):
#     print('\nInstances Labeled with ...')
#     print('will be annotated with ...')
#     for i, label in enumerate(LABELS):
#         label = label + ((11 - len(label))* ' ')
#         print('\t' + label + '\t' + str(_appraisal_ratings[i]))
#         print('\t\t\t' + str(normalized_data[i]))
#         print('\n')

# Create annotated dataset
firstline = True
with open(output_file, 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    if firstline:
        firstline = False
        output_line = ['Prior_Emotion', 'Sentence']
        for dimension in APPRAISALS:
            output_line.append(dimension)
        tsv_output.writerow(output_line)

    for i in range(len(text_instance)):
        output_line = []
        output_line.append(emotion_labels[i])
        output_line.append(text_instance[i])
        # Fill annotation
        try:
            i = LABELS.index(emotion_labels[i])
        except ValueError:
            print('\nError: Found label "%s" in the dataset. However, "%s" is not specified using the label input argument.' % (emotion_labels[i], emotion_labels[i]))
            print('Exiting.')
            sys.exit(1)

        for appraisal in APPRAISALS:
            k = APPRAISALS_ALL.index(appraisal)
            output_line.append(normalized_data[i][k])

        # Write row to file
        tsv_output.writerow(output_line)

print('\nSuccessfully created annotations in file %s.' % output_file)
