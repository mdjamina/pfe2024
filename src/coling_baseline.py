# uncompyle6 version 3.9.1
# Python bytecode version base 2.7 (62211)
# Decompiled from: Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
# Embedded file name: coling_baseline.py
# Compiled at: 2018-10-02 18:55:59
import pycrfsuite
from classifier import Classifier
from Tester import Tester
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV


class ColingBaselineClassifier(Classifier):
    crfModelName = '../models/crf_model.crfsuite'
    lmtzr = WordNetLemmatizer()

    @staticmethod
    def extractWordFeatures(tree, nodeIndex, featuresPrefix):
        word = tree[nodeIndex].word
        postag = tree[nodeIndex].pos
        return [featuresPrefix + feat for feat in [
            'word.lower=' + word.lower(),
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
            'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag))]]

    def _train(self, data):
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({'c1': 3.0,
                            'c2': 1e-20,
                            'feature.possible_transitions': True})
        x_train = map(itemgetter(0), data)
        y_train = map(itemgetter(1), data)
        trainer.append(x_train, y_train)
        trainer.train(ColingBaselineClassifier.crfModelName)

    def _classify(self, feats):
        tagger = pycrfsuite.Tagger()
        tagger.open(ColingBaselineClassifier.crfModelName)
        return tagger.tag([feats])[0]

    def _extractFeatures(self, tree, nodeIndex, candidateType):
        word = tree[nodeIndex].word
        postag = tree[nodeIndex].pos
        features = ColingBaselineClassifier.extractWordFeatures(tree, nodeIndex, '')
        if nodeIndex > 0:
            features.extend(ColingBaselineClassifier.extractWordFeatures(tree, nodeIndex - 1, '-1:'))
        else:
            features.append('BOS=true')
        if nodeIndex < len(tree) - 1:
            features.extend(ColingBaselineClassifier.extractWordFeatures(tree, nodeIndex + 1, '+1:'))
        else:
            features.append('EOS=true')
        return features


def convertPtbToLemmatizerPos(ptbPos):
    if ptbPos.startswith('VB'):
        return VERB
    if ptbPos.startswith('JJ'):
        return ADJ
    if ptbPos.startswith('RB'):
        return ADV
    return NOUN
