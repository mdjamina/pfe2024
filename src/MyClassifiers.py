from coling_baseline import ColingBaselineClassifier
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV


def set_features1(sentence, i):
    word = sentence[i][1]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        'word.length=' + str(len(word)),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isdot=%s' % isdot(word),
        'word.isdash=%s' % isdash(word),
        'word.iscomma=%s' % iscomma(word),
        'postag=' + sentence[i][0],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i][4]))
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        word1 = sentence[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.length=' + str(len(word1)),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:word.isdot=%s' % isdot(word1),
            '-1:word.isdash=%s' % isdash(word1),
            '-1:word.iscomma=%s' % iscomma(word1),
            'postag=' + sentence[i - 1][0],
            'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i - 1][4]))
        ])
    else:
        features.append('Begin_Of_Sentence')

    if i < len(sentence) - 1:
        # Set the features of relationship with next word.
        word1 = sentence[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.length=' + str(len(word1)),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:word.isdot=%s' % isdot(word1),
            '+1:word.isdash=%s' % isdash(word1),
            '+1:word.iscomma=%s' % iscomma(word1),
            'postag=' + sentence[i + 1][0],
            'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i + 1][4]))
        ])
    else:
        features.append('End_Of_Sentence')

    return features


#-----------Honnibal_CRF_Adapted---------

def set_features_honnibal_CRF_Adapted(sentence, i):
    #cB = commaBefore
    word = sentence[i][1]
    postag = sentence[i][4]
    cB = "no"
    if i > 1:
        if sentence[i - 1][1] == ',':
            cB = "yes"
    features = [
        'cB=' + cB
    ]
    return features


def set_features_honnibal(sentence, i):
    if i > 1:
        if sentence[i - 1][1] == ',':
            return "RESTR"
    return "NON-RESTR"


#-----------Dornescu---------

def set_features_dornescu(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        #'word.isupper=%s' % word.isupper(),
        #'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i - 1][1]
        postagBefore = sentence[i - 1][4]
        modifierTypeBefore = sentence[i - 1][-1]
        features.extend([
            '-1:word.lower=' + wordBefore.lower(),
            #'-1:word.isupper=%s' % wordBefore.isupper(),
            #'-1:word.istitle=%s' % wordBefore.istitle(),
            '-1postag=' + postagBefore,
            '-1postag[:2]=' + postagBefore[:2],
            '-1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagBefore)),
            '-1modifierType=' + modifierTypeBefore
        ])
    else:
        features.append('B_o_S')  #Beginning of Sentence

    if i < len(sentence) - 1:
        # Set the features of relationship with next word.
        wordAfter = sentence[i + 1][1]
        postagAfter = sentence[i + 1][4]
        modifierTypeAfter = sentence[i + 1][-1]
        features.extend([
            '+1:word.lower=' + wordAfter.lower(),
            #'+1:word.isupper=%s' % wordAfter.isupper(),
            #'+1:word.istitle=%s' % wordAfter.istitle(),
            '+1postag=' + postagAfter,
            '+1postag[:2]=' + postagAfter[:2],
            '+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
            '+1modifierType=' + modifierTypeAfter
        ])
    else:
        features.append('E_o_S')  #End of Sentence

    return features


#---------------Stanovsky-------------

def set_features_stanovsky(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    #parser = Parser()

    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        #'word.isupper=%s' % word.isupper(),
        #'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType]

    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i - 1][1]
        postagBefore = sentence[i - 1][4]
        modifierTypeBefore = sentence[i - 1][-1]
        features.extend([
            '-1:word.lower=' + wordBefore.lower(),
            #'-1:word.isupper=%s' % wordBefore.isupper(),
            #'-1:word.istitle=%s' % wordBefore.istitle(),
            '-1postag=' + postagBefore,
            '-1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagBefore)),
            '-1modifierType=' + modifierTypeBefore
        ])
    else:
        features.append('B_o_S')  #Beginning of Sentence

    if i < len(sentence) - 1:
        # Set the features of relationship with next word.
        wordAfter = sentence[i + 1][1]
        postagAfter = sentence[i + 1][4]
        modifierTypeAfter = sentence[i + 1][-1]
        features.extend([
            '+1:word.lower=' + wordAfter.lower(),
            #'+1:word.isupper=%s' % wordAfter.isupper(),
            #'+1:word.istitle=%s' % wordAfter.istitle(),
            '+1postag=' + postagAfter,
            '+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
            '+1modifierType=' + modifierTypeAfter
        ])
    else:
        features.append('E_o_S')  #End of Sentence
    '''        
    embs = loadEmbs()
    if modifierType == 'PREADJ-MOD':
        if word.lower() in embs:  
            features += ['emb{0}={1}'.format(i, emb) for (i, emb) in enumerate(embs[word.lower()])]     
    '''
    return features


def loadEmbs():
    d = {}
    with open('./Corpus/embeddedWords.txt') as fin:
        for line in fin:
            line = line.strip()
            data = line.split(' ')
            d[data[0]] = data[1:]
    return d


def set_features_unigram(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType
    ]
    return features


def set_features_unigram2(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    # Set the features of the word
    features = [
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType
    ]
    return features


def set_features_bigram(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    # Set the features of the word
    features = [
        'postag=' + postag,
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i - 1][1]
        postagBefore = sentence[i - 1][4]
        modifierTypeBefore = sentence[i - 1][-1]
        features.extend([
            '-1postag=' + postagBefore,
            '-1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagBefore)),
            '-1modifierType=' + modifierTypeBefore
        ])
    else:
        features.append('B_o_S')  #Beginning of Sentence
    return features


def set_features_novel(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]

    # Set the features of the word
    features = [
        'relativePositionInSentence=' + relativePositionInSentence(i, len(sentence)),
        'word.lower=' + word.lower(),
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        'modifierType=' + modifierType]

    if i > 0 and i < len(sentence) - 1:
        wordBefore = sentence[i - 1][1]
        postagBefore = sentence[i - 1][4]
        modifierTypeBefore = sentence[i - 1][-1]
        wordAfter = sentence[i + 1][1]
        postagAfter = sentence[i + 1][4]
        modifierTypeAfter = sentence[i + 1][-1]
        features.extend([
            '-1:word.lower|word.lower|+1:word.lower=' + wordBefore.lower() + '|' + word.lower() + '|' + wordAfter.lower(),
            '-1postag|postag|+1postag=' + postagBefore + '|' + postag + '|' + postagAfter,
            '-1lemma|lemma|+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(
                postagBefore)) + '|' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(
                postag)) + '|' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
            '-1modifierType|modifierType|+1modifierType=' + modifierTypeBefore + '|' + modifierType + '|' + modifierTypeAfter,
        ])
    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i - 1][1]
        postagBefore = sentence[i - 1][4]
        modifierTypeBefore = sentence[i - 1][-1]
        features.extend([
            '-1:word.lower|word.lower=' + wordBefore.lower() + '|' + word.lower(),
            '-1postag|postag=' + postagBefore + '|' + postag,
            '-1postag[:2]|postag[:2]=' + postagBefore[:2] + '|' + postag[:2],
            '-1lemma|lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(
                postagBefore)) + '|' + ColingBaselineClassifier.lmtzr.lemmatize(word,
                                                                                convertPtbToLemmatizerPos(postag)),
            '-1modifierType|modifierType=' + modifierTypeBefore + '|' + modifierType,
            '-1type|-1postag|type|postag=' + modifierTypeBefore + '|' + postagBefore + '|' + modifierType + postag
        ])
    else:
        features.append('B_o_S')  #Beginning of Sentence

    if i < len(sentence) - 1:
        # Set the features of relationship with next word.
        wordAfter = sentence[i + 1][1]
        postagAfter = sentence[i + 1][4]
        modifierTypeAfter = sentence[i + 1][-1]
        features.extend([
            'word.lower|+1:word.lower= ' + word.lower() + '|' + wordAfter.lower(),
            'postag|+1postag=' + postag + '|' + postagAfter,
            'postag[:2]|+1postag[:2]=' + postag + '|' + postagAfter[:2],
            'lemma|+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(
                postag)) + '|' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
            'modifierType|+1modifierType=' + modifierType + '|' + modifierTypeAfter,
            'type|postag|+1type|+1postag=' + modifierType + '|' + postag + '|' + modifierTypeAfter + "|" + postagAfter,
        ])
    else:
        features.append('E_o_S')  #End of Sentence

    if modifierType == 'PREADJ-MOD' or modifierType == 'PP-MOD':
        features.append('relativePositionInSentence=' + relativePositionInSentence(i, len(sentence)))

    '''
    embs = loadEmbs()
    if modifierType == 'PREADJ-MOD':
        print(modifierType)
        if word.lower() in embs:  
            features += ['emb{0}={1}'.format(i, emb) for (i, emb) in enumerate(embs[word.lower()])]     
    '''
    return features


def relativePositionInSentence(wordPos, lengthOfSentence):
    if float(wordPos) / lengthOfSentence < 0.25:
        return "1"  #first Quadrant
    elif float(wordPos) / lengthOfSentence < 0.50:
        return "2"  #second Quadrant
    elif float(wordPos) / lengthOfSentence < 0.75:
        return "3"  #second Quadrant
    else:
        return "4"  #fourth Quadrant


def isdot(word):
    return True if word in '.' else False


def isdash(word):
    return True if word in '-' else False


def iscomma(word):
    return True if word in ',' else False


def get_features(sent, type):
    if type == 1:
        return [set_features1(sent, i) for i in range(len(sent))]
    elif type == "dornescu":
        return [set_features_dornescu(sent, i) for i in range(len(sent))]
    elif type == "honnibal":
        return [set_features_honnibal(sent, i) for i in range(len(sent))]
    elif type == "honnibal":
        return [set_features_honnibal_CRF_Adapted(sent, i) for i in range(len(sent))]
    elif type == "unigram":
        return [set_features_unigram(sent, i) for i in range(len(sent))]
    elif type == "bigram":
        return [set_features_bigram(sent, i) for i in range(len(sent))]
    elif type == "novel":
        return [set_features_novel(sent, i) for i in range(len(sent))]
    elif type == "stanovsky":
        return [set_features_stanovsky(sent, i) for i in range(len(sent))]
    else:
        raise Exception


def get_labels(sent):
    return [x[-2] for x in sent]


def get_tokens(sent):
    return [x[-2] for x in sent]


def convertPtbToLemmatizerPos(ptbPos):
    if ptbPos.startswith('VB'):
        return VERB
    if ptbPos.startswith('JJ'):
        return ADJ
    if ptbPos.startswith('RB'):
        return ADV
    return NOUN
