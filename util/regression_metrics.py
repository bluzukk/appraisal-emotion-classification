#!/usr/bin/env python
"""
Helper for evaluating multiclass regression

"""
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error

class metrics_regression:
    def __init__(self, LABELS, decimals):
        # self.mse = [0] * len(LABELS)
        self.mae = [0] * len(LABELS)

        maxlen = 20
        self.LABELS = [((maxlen - len(x))* ' ') + x for x in LABELS]
        self.DECIMALS = decimals
        self.folds = 0

    def evaluateFold(self, predicted, correct):
        # Add and print results of a fold
        mse = [0] * len(self.LABELS)
        mae = [0] * len(self.LABELS)

        correct = correct.astype(float)
        predicted = predicted.astype(float)

        mean_abs_error = [0] * len(self.LABELS)
        instances_count = len(correct)
        for i in range(instances_count):
            for p in range(len(correct[i])):
                error = predicted[i][p] - correct[i][p]
                mean_abs_error[p] += abs(error) / instances_count
                # print('\nPredicted %f' % predicted[i][p])
                # print('Correct %f' % correct[i][p])
                # print('Error %f' % error)

        print('\n          Appraisal \t MAE')
        print('-'*40)
        for i, label in enumerate(self.LABELS):
            print('%s \t %.*f' % (label, self.DECIMALS, mean_abs_error[i]))
        print('-'*40)
        total_abs_error = sum(mean_abs_error) / len(self.LABELS)
        print('          Total \t %.*f' % (self.DECIMALS, total_abs_error))

        # average results over all folds
        self.mae = [sum(x) for x in zip(mean_abs_error, self.mae)]
        self.folds += 1

    def showResults(self):
        print(self.mae)
        print('\n          Appraisal \t MAE')
        print('-'*40)
        for i, label in enumerate(self.LABELS):
            mae = self.mae[i] / self.folds
            print('%s \t %.*f' % (label, self.DECIMALS, mae))
        print('-'*40)
        total_abs_error = sum(self.mae) / len(self.LABELS) / self.folds
        print('          Total \t %.*f' % (self.DECIMALS, total_abs_error))
