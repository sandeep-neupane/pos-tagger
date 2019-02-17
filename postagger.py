#!/usr/bin/env python
#Author: Sandeep Neupane

from optparse import OptionParser
import os, logging
import utils
import collections
import operator

def create_model(sentences):
    model = collections.defaultdict(lambda: collections.defaultdict(int))
    for sentence in sentences:
        for token in sentence:
            model[token.word][token.tag] +=1 #increasing the counter
    return model


def predict_tags(sentences, model):
    for sentence in sentences:
        for token in sentence:
            dictForWord  = model.get(token.word)
            if(dictForWord != None):
                POS_with_max_value = max(dictForWord.iteritems(), key=operator.itemgetter(1))[0]
                token.tag = POS_with_max_value
            else:
                #there can be another method to it like  assignning most frequent tag in training set
                token.tag = 'NN'
    return sentences

if __name__ == "__main__":
    usage = "usage: %prog [options] GOLD TEST"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--debug", action="store_true",
                      help="turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Please provide required arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    training_file = args[0]
    training_sents = utils.read_tokens(training_file)
    test_file = args[1]
    test_sents = utils.read_tokens(test_file)

    model = create_model(training_sents)

    ## read sentences again because predict_tags(...) rewrites the tags
    sents = utils.read_tokens(training_file)
    predictions = predict_tags(sents, model)
    accuracy = utils.calc_accuracy(training_sents, predictions)
    print "Accuracy in training [%s sentences]: %s" % (len(sents), accuracy)

    ## read sentences again because predict_tags(...) rewrites the tags
    sents = utils.read_tokens(test_file)
    predictions = predict_tags(sents, model)
    accuracy = utils.calc_accuracy(test_sents, predictions)
    print "Accuracy in testing [%s sentences]: %s" % (len(sents), accuracy)
