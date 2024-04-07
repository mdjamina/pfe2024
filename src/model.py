#!/usr/bin/env python
# coding: utf-8

'''
Model to train for differentiating restrictive vs. non-restrictive

@author: Mustafa Bal
'''

import sys, time, string
from tabulate import tabulate
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

import pycrfsuite
import MyClassifiers


def get_input(input_file):
    '''
    get input file by utf8 encoding. Read the file content then return it.
    '''
    sentence = []
    result_list = []

    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            #(0)posInSentence
            #(1)spelling1, (2)spelling2, (3)spelling3
            #(4)postype1, (5)postype2
            #(6)unk1, (7)unk2
            #(8)connected1, (9)connected2,  
            #(10)candidateType1, (11)candidateType2
            #(-2)RESTR/NON-RESTR, (-1)modifierType

            if data == ['']:  # end of sentence
                result_list.append(sentence)
                sentence = []
            elif data[-2] != "_":
                tup = (
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], int(data[8]), int(data[9]),
                data[-2], data[-1])
                sentence.append(tup)
        f.close()
    return result_list


def bio_classification_report(y_true, y_pred):
    '''
    Classification report for a list of BIO-encoded sequences.    
    Note that it requires scikit-learn 0.15+ (or a version from github master) to calculate averages properly!
    '''
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_)

    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def create_report(y_true, y_pred):
    tagset = set(["APPOS-MOD", "INF-MOD", "PP-MOD", "PREADJ-MOD", "PREVERB-MOD", "RC-MOD", "POSTADJ-MOD"])
    #total, TruePos, TrueNeg, FalsPos, FalsNeg
    #Pos: Non-restrictive modification
    #Neg: Restrictive modification
    d = {el: [0.0, 0.0, 0.0, 0.0, 0.0] for el in tagset}
    blockC = 0
    wordC = 0
    with open(testing_file) as f:
        for line in f:
            if line != "\n":
                lineLst = line.split("\t")
                if lineLst[-1] != "_\n":
                    modifierType = lineLst[-1].replace('\n', '')
                    d[modifierType][0] += 1
                    if y_true[blockC][wordC] == y_pred[blockC][wordC]:
                        if y_true[blockC][wordC] == "NON-RESTR":
                            d[modifierType][1] += 1
                        else:
                            d[modifierType][2] += 1
                    else:
                        if y_true[blockC][wordC] == "NON-RESTR":
                            d[modifierType][4] += 1
                        else:
                            d[modifierType][3] += 1
                else:
                    wordC += 1
            else:
                blockC += 1
                wordC = 0
    return tabulate(
        getScoreData(tagset, d),
        headers=['Modifier Type', '#', 'Accuracy', 'Precision', 'Recall', 'F1'])


def getScoreData(tagset, d):
    # d[tag][1]: true positive - pred: NR, actual: NR
    # d[tag][2]: true negative - pred: R, actual:R
    # d[tag][3]: false positive - pred: NR, actual: R
    # d[tag][4]: false negative - pred: R, actual: NR
    returnLst = []
    tTP = 0.0
    tTN = 0.0
    tFP = 0.0
    tFN = 0.0
    tT = 0.0
    #for each tag
    for tag in tagset - {"POSTADJ-MOD"}:
        if d[tag][1] + d[tag][3] != 0:
            precision = d[tag][1] / (d[tag][1] + d[tag][3])
        else:
            precision = 0
        if d[tag][1] + d[tag][4] != 0:
            recall = d[tag][1] / (d[tag][1] + d[tag][4])
        else:
            recall = 0
        accuracy = (d[tag][1] + d[tag][2]) / (d[tag][0])
        if recall + precision != 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0
        tT += d[tag][0]
        tTP += d[tag][1]
        tTN += d[tag][2]
        tFP += d[tag][3]
        tFN += d[tag][4]
        returnLst.append([tag, d[tag][0], round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)])

    #total
    if tTP + tFP != 0:
        totalPrecision = tTP / (tTP + tFP)
    else:
        totalPrecision = 0
    totalRecall = tTP / (tTP + tFN)
    totalAccuracy = (tTP + tTN) / (tT)
    if totalRecall + totalPrecision != 0:
        totalF1 = 2 * (totalRecall * totalPrecision) / (totalRecall + totalPrecision)
    else:
        totalF1 = 0
    returnLst.append(
        ["Total", int(tT), round(totalAccuracy, 2), round(totalPrecision, 2), round(totalRecall, 2), round(totalF1, 2)])
    return returnLst


def getTrainerFeatures(ft):
    #c1: coefficient for L1 penalty
    #c2: coefficient for L2 penalty
    #'feature.possible_transitions':
    #   include transitions that are possible, but not observed
    """ if ft == "dornescu" or ft == "honnibal":
        return {'c1': 3.0,'c2': 1e-20,'feature.possible_transitions': True}
    elif ft == 1 or ft == 2:
        return {'c1': 0.5,'c2': 1e-3, 'max_iterations': 1000, 'feature.possible_transitions': True}
    else:
        raise Exception """
    return {'c1': 0.5, 'c2': 1e-3, 'max_iterations': 1000, 'feature.possible_transitions': True}  #Own version
    #return {'c1': 3.0,'c2': 1e-20,'feature.possible_transitions': True} #Stanovsky version


def main(training_file, testing_file, model_file, ft):
    start = time.time()

    # Get training and testing set of data
    #Ignoring the ones where RESTR/NON-RESTR is not defined, i.e. the ones written as "_"
    training_set = get_input(training_file)
    testing_set = get_input(testing_file)

    #Special training case for Honnibal et al.
    if ft == "honnibal":
        y_test = [MyClassifiers.get_labels(s) for s in testing_set]
        y_pred = [MyClassifiers.get_features(s, ft) for s in testing_set]
        printReport(y_pred, y_test, ft)
        return

    # Get features of each word on training set
    X_train = []
    count = 0
    for s in training_set:
        X_train.append(MyClassifiers.get_features(s, ft))
        count += 1
    y_train = [MyClassifiers.get_labels(s) for s in training_set]

    # Get features of each word on testing set
    X_test = [MyClassifiers.get_features(s, ft) for s in testing_set]
    y_test = [MyClassifiers.get_labels(s) for s in testing_set]

    # Create trainer model of CRF
    trainer = pycrfsuite.Trainer(verbose=False)

    trainer.set_params(getTrainerFeatures(ft))

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Train the model and save the trained model into model_file
    trainer.train(model_file_loc + ft + "Model.crfsuite")
    #print ("Log of last iteration={}".format(trainer.logparser.iterations[-1]))

    # Initial tagger for prediction task
    trained_model = pycrfsuite.Tagger()
    trained_model.open(model_file_loc + ft + "Model.crfsuite")  # Load the trained model.
    # Get prediction tag results from trained model
    y_pred = [trained_model.tag(xseq) for xseq in X_test]

    # Print the Precision, Recall, and F-1 score
    end = time.time()
    print('Runtime:', end - start)
    printReport(y_test, y_pred, ft)


def printReport(y_test, y_pred, ft):
    print('-----------------------------------------------------------------------')
    print('CRF model has been generated: ' + ft)
    print(create_report(y_test, y_pred))
    #print(bio_classification_report(y_test, y_pred))
    print('-----------------------------------------------------------------------')
    print('')


training_file = "corpus/combined_train_and_dev.txt"
testing_file = "corpus/test.txt"
model_file_loc = "models/"

#main(training_file, testing_file, model_file_loc, "unigram")
#main(training_file, testing_file, model_file_loc, "bigram")
main(training_file, testing_file, model_file_loc, "novel")
main(training_file, testing_file, model_file_loc, "stanovsky")
main(training_file, testing_file, model_file_loc, "dornescu")
