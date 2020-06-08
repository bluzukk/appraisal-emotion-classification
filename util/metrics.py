#!/usr/bin/env python
"""
Helper for evaluating classification predictions

- Note that all metrics are currently based on the
confusion matrix impementation from sklearn.
- Can only handle multi-class classifications

"""

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

class metrics:
    def __init__(self, classes_true, classes_predicted, CLASSES, decimals):
        # Pad label strings
        maxlen = len(max(CLASSES, key=len))
        if (maxlen < 10):
            maxlen = 10
        self.CLASSES = [((maxlen - len(x))* ' ') + x for x in CLASSES]
        self.decimals = decimals
        self.confusion_matrix = [[0] * len(CLASSES) for _ in range(len(CLASSES))]

        if ((classes_true is not None) and (classes_predicted is not None)):
            self.confusion_matrix = confusion_matrix(classes_true, classes_predicted)

    def addIntermediateResults(self, y_correct, y_predict):
        """
        Adds results of a fold or round to the total results
        """
        cm = confusion_matrix(y_correct, y_predict)
        self.confusion_matrix = np.add(cm, self.confusion_matrix)

    def calculateTruePositives(self):
        truePositives = np.diag(self.confusion_matrix)
        return truePositives

    def calculateFalsePositives(self):
        truePositives = np.diag(self.confusion_matrix)
        falsePositives = np.sum(self.confusion_matrix, axis=0) - truePositives
        return falsePositives

    def calculateFalseNegatives(self):
        truePositives = np.diag(self.confusion_matrix)
        falseNegatives = np.sum(self.confusion_matrix, axis=1) - truePositives
        return falseNegatives

    def calculatePrecision(self):
        """
        Calculate precision for every class
        P = tp / (tp + fp)
        """
        truePositives = np.diag(self.confusion_matrix)
        falsePositives = np.sum(self.confusion_matrix, axis=0) - truePositives
        precisions = []
        for i in range(len(self.CLASSES)):
            if (truePositives[i] != 0 and truePositives[i] + falsePositives[i] != 0):
                precision = truePositives[i] / (truePositives[i] + falsePositives[i])
                precisions.append(precision)
            else:
                precisions.append(0)
        return precisions

    def calculateRecall(self):
        """
        Calculate recall for every class
        R = tp / (tp + fn)
        """
        truePositives = np.diag(self.confusion_matrix)
        falseNegatives = np.sum(self.confusion_matrix, axis=1) - truePositives
        recalls = []
        for i in range(len(self.CLASSES)):
            if (truePositives[i] != 0 and truePositives[i] + falseNegatives[i] != 0):
                recall = truePositives[i] / (truePositives[i] + falseNegatives[i])
                recalls.append(recall)
            else:
                recalls.append(0)
        return recalls

    def calculateFScore(self):
        """
        Calculate F1 score for every class
        F_1 = 2 * ((p*r) / (p+r))
        """
        precisions = self.calculatePrecision()
        recalls = self.calculateRecall()
        f1_scores = []
        for i in range(len(self.CLASSES)):
            if (precisions[i] != 0 and recalls[i] != 0):
                f1 = 2 * ((precisions[i] * recalls[i]) / (precisions[i] + recalls[i]))
                f1_scores.append(f1)
            else:
                f1_scores.append(0)
        return f1_scores

    def calculatePMicro(self):
        tp = sum(self.calculateTruePositives())
        fp = sum(self.calculateFalsePositives())
        p_micro = tp / (tp + fp)
        p_micro = round(p_micro, 10)
        return p_micro

    def calculateRMicro(self):
        tp = sum(self.calculateTruePositives())
        fn = sum(self.calculateFalseNegatives())
        r_micro = tp / (tp + fn)
        r_micro = round(r_micro, 10)
        return r_micro

    def calculateFMicro(self):
        """
        Calculate F1 score for every class
        F_1 = 2 * ((p*r) / (p+r))
        """
        tp = sum(self.calculateTruePositives())
        fp = sum(self.calculateFalsePositives())
        fn = sum(self.calculateFalseNegatives())
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1_micro = 2 * ((p*r) / (p+r))
        f1_micro = round(f1_micro, 10)
        return f1_micro

    def showResults(self):
        """
        Show classification results in a table
        """
        tab = '\t'

        print('\t\tTP\tFP\tFN\tPrec.\tRec.\tF1')
        print('-'*60)
        tp = self.calculateTruePositives()
        fp = self.calculateFalsePositives()
        fn = self.calculateFalseNegatives()
        precisions = self.calculatePrecision()
        recalls = self.calculateRecall()
        f1_scores = self.calculateFScore()

        # Calculate class specific results
        # Iterate over classes and print informations str()
        for i in range(len(self.CLASSES)):
            # Round and print metrics for each class
            precision = '%.*f' % (self.decimals, precisions[i])
            recall = '%.*f' % (self.decimals, recalls[i])
            f1_score = '%.*f' % (self.decimals, f1_scores[i])
            result = self.CLASSES[i] + tab*1
            result = result + str(tp[i]) + tab + str(fp[i]) + tab + str(fn[i]) + tab
            result = result + str(precision) + tab
            result = result + str(recall) + tab
            result = result + str(f1_score)
            print(result)

        # Calulate total results
        tp_total = str(sum(tp))
        fp_total = str(sum(fp))
        fn_total = str(sum(fn))
        precision = str(round(sum(precisions) / len(self.CLASSES), self.decimals))
        recall = str(round(sum(recalls) / len(self.CLASSES), self.decimals))
        f1_macro = str(round(sum(f1_scores) / len(self.CLASSES), self.decimals))

        tp = sum(tp)
        fp = sum(fp)
        fn = sum(fn)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1_micro = 2 * ((p*r) / (p+r))
        f1_micro = str(round(f1_micro, 2))

        # f1_macro = 2 * ((sum(precisions) / len(self.CLASSES) * (sum(recalls) / len(self.CLASSES)) / ((sum(precisions) / len(self.CLASSES) + (sum(recalls) / len(self.CLASSES))))))
        # f1_macro = str(f1_macro)

        total = 'Total:' + tab*2
        total = total + tp_total + tab + fp_total + tab + fn_total + tab
        total = total + precision + tab
        total = total + recall + tab
        total = total + f1_macro + '(micro:' + f1_micro + ')'+ tab
        # total = total + f1_macro
        print('-'*60)
        print(total)

    def showConfusionMatrix(self, do_plot):
        if do_plot:
            print('Implement me')
            import seaborn as sn
            import matplotlib.pyplot as plt
            df_cm = pd.DataFrame(self.confusion_matrix, self.CLASSES, self.CLASSES)
            #plt.figure(figsize = (10,7))
            sn.set(font_scale=1.4)#for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
            plt.show()
        else:
            print(self.confusion_matrix)

        return self.confusion_matrix


    def createLatexResults(self):
        """
        Creates results in latex format
        """
        _and = ' & '
        line_end = '\\\\'

        print('\\toprule')
        print('& TP & FP & FN & Precision. & Recall. & F1 \\\\')
        print('\\toprule')
        tp = self.calculateTruePositives()
        fp = self.calculateFalsePositives()
        fn = self.calculateFalseNegatives()
        precisions = self.calculatePrecision()
        recalls = self.calculateRecall()
        f1_scores = self.calculateFScore()

        # Calculate class specific results
        # Iterate over classes and print informations str()
        for i in range(len(self.CLASSES)):
            # Round and print metrics for each class
            precision = '%.*f' % (self.decimals, precisions[i])
            recall = '%.*f' % (self.decimals, recalls[i])
            f1_score = '%.*f' % (self.decimals, f1_scores[i])
            result = self.CLASSES[i] + _and
            result = result + str(tp[i]) + _and + str(fp[i]) + _and + str(fn[i]) + _and
            result = result + str(precision) + _and
            result = result + str(recall) + _and
            result = result + str(f1_score) + line_end
            print(result)

        # Calulate total results
        tp_total = str(sum(tp))
        fp_total = str(sum(fp))
        fn_total = str(sum(fn))
        precision = str(round(sum(precisions) / len(self.CLASSES), self.decimals))
        recall = str(round(sum(recalls) / len(self.CLASSES), self.decimals))
        f1_macro = str(round(sum(f1_scores) / len(self.CLASSES), self.decimals))

        # f1_macro = 2 * ((sum(precisions) / len(self.CLASSES) * (sum(recalls) / len(self.CLASSES)) / ((sum(precisions) / len(self.CLASSES) + (sum(recalls) / len(self.CLASSES))))))
        # f1_macro = str(f1_macro)

        print('\\midrule')
        total = 'Total:' + _and
        total = total + tp_total + _and + fp_total + _and + fn_total + _and
        total = total + precision + _and
        total = total + recall + _and
        total = total + f1_macro + line_end
        # total = total + f1_macro
        print(total)
        print('\\bottomrule')

    def createMarkdownResults(self):
        """Creates results in markdown format
        """
        sep = ' | '
        align_left = ' :--- '
        center = ' :----: '

        print('| Emotion | TP | FP | FN | Precision. | Recall. | F1 |')
        print(sep + align_left + (sep + center)*6 + sep )
        tp = self.calculateTruePositives()
        fp = self.calculateFalsePositives()
        fn = self.calculateFalseNegatives()
        precisions = self.calculatePrecision()
        recalls = self.calculateRecall()
        f1_scores = self.calculateFScore()

        # Calculate class specific results
        # Iterate over classes and print informations str()
        for i in range(len(self.CLASSES)):
            # Round and print metrics for each class
            precision = '%.*f' % (self.decimals, precisions[i])
            recall = '%.*f' % (self.decimals, recalls[i])
            f1_score = '%.*f' % (self.decimals, f1_scores[i])
            result = sep + self.CLASSES[i] + sep
            result = result + str(tp[i]) + sep + str(fp[i]) + sep + str(fn[i]) + sep
            result = result + str(precision) + sep
            result = result + str(recall) + sep
            result = result + str(f1_score) + sep
            print(result)

        # Calulate total results
        tp_total = str(sum(tp))
        fp_total = str(sum(fp))
        fn_total = str(sum(fn))
        precision = str(round(sum(precisions) / len(self.CLASSES), self.decimals))
        recall = str(round(sum(recalls) / len(self.CLASSES), self.decimals))
        f1_macro = str(round(sum(f1_scores) / len(self.CLASSES), self.decimals))

        total = sep + 'Total:' + sep
        total = total + tp_total + sep + fp_total + sep + fn_total + sep
        total = total + precision + sep
        total = total + recall + sep
        total = total + f1_macro + sep
        # total = total + f1_macro
        print(total)


    def writeResults(self, EXPERIMENTNAME, SAVEFILE):
        """
        Write classification results to a file
        """
        tab = '\t'
        with open(SAVEFILE, "a") as myfile:
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M')
            myfile.write('\n\n\n')
            myfile.write('\n' + now.strftime('%Y-%m-%d %H:%M'))
            myfile.write('\n'         + EXPERIMENTNAME + '\n\n')
            myfile.write('\t\tTP\tFP\tFN\tPrec.\tRec.\tF1')
            myfile.write('\n\n')
            myfile.write('-'*60)
            print('\t\tTP\tFP\tFN\tPrec.\tRec.\tF1')
            print('-'*60)
            myfile.write('\n')
            tp = self.calculateTruePositives()
            fp = self.calculateFalsePositives()
            fn = self.calculateFalseNegatives()
            precisions = self.calculatePrecision()
            recalls = self.calculateRecall()
            f1_scores = self.calculateFScore()

            # print(self.confusion_matrix)

            # Calculate class specific results
            # Iterate over classes and print informations str()
            for i in range(len(self.CLASSES)):
                # Round and print metrics for each class
                precision = round(precisions[i], self.decimals)
                recall = round(recalls[i], self.decimals)
                f1_score = round(f1_scores[i], self.decimals)
                result = self.CLASSES[i] + tab
                result = result + str(tp[i]) + tab + str(fp[i]) + tab + str(fn[i]) + tab
                result = result + str(precision) + tab
                result = result + str(recall) + tab
                result = result + str(f1_score)
                print(result)
                myfile.write(result+'\n')

            # Calulate total results
            tp_total = str(sum(tp))
            fp_total = str(sum(fp))
            fn_total = str(sum(fn))
            precision = str(round(sum(precisions) / len(self.CLASSES), self.decimals))
            recall = str(round(sum(recalls) / len(self.CLASSES), self.decimals))
            f1_macro = str(round(sum(f1_scores) / len(self.CLASSES), self.decimals))

            tp = sum(tp)
            fp = sum(fp)
            fn = sum(fn)
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1_micro = 2 * ((p*r) / (p+r))
            f1_micro = str(round(f1_micro, 2))

            # f1_macro = 2 * ((sum(precisions) / len(self.CLASSES) * (sum(recalls) / len(self.CLASSES)) / ((sum(precisions) / len(self.CLASSES) + (sum(recalls) / len(self.CLASSES))))))
            # f1_macro = str(f1_macro)

            total = 'Total:' + tab*2
            total = total + tp_total + tab + fp_total + tab + fn_total + tab
            total = total + precision + tab
            total = total + recall + tab
            total = total + f1_macro + '(micro:' + f1_micro + ')'+ tab
            # total = total + f1_macro
            print('-'*60)
            myfile.write('-'*60)
            print(total)
            myfile.write('\n' + total)
            myfile.write('\n')
            myfile.write(str(self.confusion_matrix))
            myfile.write('\n')
        myfile.close()
