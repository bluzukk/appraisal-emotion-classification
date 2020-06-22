#!/usr/bin/env python
"""
Helper for evaluating multilabel classification predictions

"""

class metrics:
    def __init__(self, LABELS, decimals):
        self.tp_sum = [0] * len(LABELS)
        self.fp_sum = [0] * len(LABELS)
        self.fn_sum = [0] * len(LABELS)

        # Pad label strings
        maxlen = 20
        self.LABELS = [((maxlen - len(x))* ' ') + x for x in LABELS]
        # The number of decimals shown
        self.decimals = decimals

    def evaluateFold(self, predicted, correct):
        # Add and print results of a fold
        tp = [0] * len(self.LABELS)
        fp = [0] * len(self.LABELS)
        fn = [0] * len(self.LABELS)

        for index in range(len(predicted)):
            for i in range(len(self.LABELS)):
                # print(results[index])
                # print(ohe_test[index])
                if predicted[index][i] == 1 and correct[index][i] == 1:
                    tp[i] += 1
                if predicted[index][i] == 0 and correct[index][i] == 1:
                    fn[i] += 1
                if predicted[index][i] == 1 and correct[index][i] == 0:
                    fp[i] += 1

        self.showResults(tp, fp, fn)
        self.tp_sum = [sum(x) for x in zip(tp, self.tp_sum)]
        self.fp_sum = [sum(x) for x in zip(fp, self.fp_sum)]
        self.fn_sum = [sum(x) for x in zip(fn, self.fn_sum)]

    def calculateMetrics(self, truePositives, falsePositives, falseNegatives):
        precision = 0
        if (truePositives != 0 and truePositives + falsePositives != 0):
            precision = truePositives / (truePositives + falsePositives)
        recall = 0
        if (truePositives != 0 and truePositives + falseNegatives != 0):
            recall = truePositives / (truePositives + falseNegatives)
        f1 = 0
        if (precision != 0 and recall != 0):
            f1 = 2 * ((precision * recall) / (precision + recall))
        return precision, recall, f1


    def showFinalResults(self, format):
        """
        Show classification results
        """
        tab = '\t'
        sep = ' | '
        sep_ = '| '
        align_left = ' :--- '
        center = ' :----: '
        _and = ' & '
        line_end = '\\\\'


        tp = self.tp_sum
        fp = self.fp_sum
        fn = self.fn_sum

        if (format == 'text'):
            print('\t\t\tTP\tFP\tFN\tPrec.\tRec.\tF1')
            print('-'*70)
        elif (format == 'latex'):
            print('\\toprule')
            print('& TP & FP & FN & Precision. & Recall. & F1 \\\\')
            print('\\toprule')
        elif (format == 'markdown'):
            print('| Emotion | TP | FP | FN | Precision. | Recall. | F1 |')
            print(sep + align_left + (sep + center)*6 + sep )

        # Calculate class specific results
        precisions = [0] * len(self.LABELS)
        recalls = [0] * len(self.LABELS)
        f1_scores = [0] * len(self.LABELS)
        for i in range(len(self.LABELS)):
            # Round and print metrics for each class
            precision, recall, f1_score = self.calculateMetrics(tp[i], fp[i], fn[i])
            precisions[i] = precision
            recalls[i] = recall
            f1_scores[i] = f1_score
            precision = '%.*f' % (self.decimals, precision)
            recall = '%.*f' % (self.decimals, recall)
            f1_score = '%.*f' % (self.decimals, f1_score)
            if (format == 'latex'):
                result = self.LABELS[i] + _and
                result = result + str(tp[i]) + _and + str(fp[i]) + _and + str(fn[i]) + _and
                result = result + str(precision) + _and
                result = result + str(recall) + _and
                result = result + str(f1_score) + _and
            elif (format == 'markdown'):
                result = sep_ + self.LABELS[i] + sep
                result = result + str(tp[i]) + sep + str(fp[i]) + sep + str(fn[i]) + sep
                result = result + str(precision) + sep
                result = result + str(recall) + sep
                result = result + str(f1_score) + sep
            else:
                result = self.LABELS[i] + tab*2
                result = result + str(tp[i]) + tab + str(fp[i]) + tab + str(fn[i]) + tab
                result = result + str(precision) + tab
                result = result + str(recall) + tab
                result = result + str(f1_score)
            print(result)

        precision = str(round(sum(precisions) / len(self.LABELS), self.decimals))
        recall = str(round(sum(recalls) / len(self.LABELS), self.decimals))
        f1_macro = str(round(sum(f1_scores) / len(self.LABELS), self.decimals))

        # Calulate total results
        tp_total = sum(tp)
        fp_total = sum(fp)
        fn_total = sum(fn)

        # Micro scores
        p = tp_total / (tp_total + fp_total)
        r = tp_total / (tp_total + fn_total)
        f1_micro = 2 * ((p*r) / (p+r))
        f1_micro = str(round(f1_micro, 2))

        if (format == 'latex'):
            print('\\midrule')
            total = '\tTotal' + tab*2
            total = total + str(tp_total) + _and + str(fp_total) + _and + str(fn_total) + _and
            total = total + precision + _and
            total = total + recall + _and
            total = total + f1_macro + '(micro:' + f1_micro + ')'+ line_end
        if (format == 'markdown'):
            total = sep_ + 'Total' + sep
            total = total + str(tp_total) + sep + str(fp_total) + sep + str(fn_total) + sep
            total = total + precision + sep
            total = total + recall + sep
            total = total + f1_macro + '(micro:' + f1_micro + ')'+ sep
        else:
            print('-'*70)
            total = '\tTotal' + tab*2
            total = total + str(tp_total) + tab + str(fp_total) + tab + str(fn_total) + tab
            total = total + precision + tab
            total = total + recall + tab
            total = total + f1_macro + '(micro:' + f1_micro + ')'+ tab
        print(total)


    def showResults(self, tp, fp, fn):
        """
        Show classification results in a table
        """
        tab = '\t'

        print('\t\t\tTP\tFP\tFN\tPrec.\tRec.\tF1')
        print('-'*70)

        # Calculate class specific results
        precisions = [0] * len(self.LABELS)
        recalls = [0] * len(self.LABELS)
        f1_scores = [0] * len(self.LABELS)
        for i in range(len(self.LABELS)):
            # Round and print metrics for each class
            precision, recall, f1_score = self.calculateMetrics(tp[i], fp[i], fn[i])
            precisions[i] = precision
            recalls[i] = recall
            f1_scores[i] = f1_score
            precision = '%.*f' % (self.decimals, precision)
            recall = '%.*f' % (self.decimals, recall)
            f1_score = '%.*f' % (self.decimals, f1_score)
            result = self.LABELS[i] + tab
            result = result + str(tp[i]) + tab + str(fp[i]) + tab + str(fn[i]) + tab
            result = result + str(precision) + tab
            result = result + str(recall) + tab
            result = result + str(f1_score)
            print(result)

        # Macro scores
        precision = sum(precisions) / len(self.LABELS)
        recall = sum(recalls) / len(self.LABELS)
        f1_macro = sum(f1_scores) / len(self.LABELS)

        # Micro scores
        tp_total = sum(tp)
        fp_total = sum(fp)
        fn_total = sum(fn)
        print('-'*70)
        p_micro = tp_total / (tp_total + fp_total)
        r_micro = tp_total / (tp_total + fn_total)
        f1_micro = 2 * ((p_micro*r_micro) / (p_micro+r_micro))

        print('\tTotal (macro)\t%d\t%d\t%d\t%.*f\t%.*f\t%.*f' %
                        (tp_total, fp_total, fn_total,
                        self.decimals, precision,
                        self.decimals, recall,
                        self.decimals, f1_macro))

        print('\tTotal (micro)\t\t\t\t%.*f\t%.*f\t%.*f' % (
                        self.decimals, p_micro,
                        self.decimals, r_micro,
                        self.decimals, f1_micro))
