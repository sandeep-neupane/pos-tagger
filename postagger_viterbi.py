#!/usr/bin/env python
#Author: Sandeep Neupane
from optparse import OptionParser
import utils
import collections
import logging,sys


def calculateWordTagCOunt(sentences):
	wordTagCount = collections.defaultdict(lambda: collections.defaultdict(int))
    tagTagCount = collections.defaultdict(lambda: collections.defaultdict(int))
    tagCount = collections.defaultdict(int)
    for sentence in sentences:
        previousToken = "<s>"
        for token in sentence:
            wordTagCount[token.word][token.tag] += 1
            tagCount[token.tag] += 1
            tagTagCount[previousToken][token.tag]+=1
            previousToken = token.tag
    totalNumOfTokens = sum(tagCount.values()) #this should be done before below step
    tagCount["<s>"] = len(sentences)
    return [wordTagCount,tagTagCount,tagCount,totalNumOfTokens]



def calculateWordTagProb(tagCount,wordTagCount):
    wordTagProb = collections.defaultdict(lambda: collections.defaultdict(float))
    #removing <"s"> from listOftags -> we dont want to predict this :D
    tags = (tagCount.keys())
    tags.remove("<s>")
    for word in wordTagCount.keys():
        for tag in tags:
            numerator = wordTagCount[word][tag]
            denominator = tagCount[tag]
            wordTagProb[word][tag] = float(numerator)/(denominator)
    return wordTagProb

def calculateTagTagProb(tagTagCount,tagCount):
    tagTagProb = collections.defaultdict(lambda: collections.defaultdict(float))
    for tag in tagTagCount.keys():
        for anotherTag in tagTagCount.keys():
            numerator = tagTagCount[tag][anotherTag]
            denominator = tagCount[tag]
            tagTagProb[anotherTag][tag] = float(numerator+1)/(denominator+len(tagCount))
    return tagTagProb



def create_model(sentences):
    print("Creating Model")
    wordTagCount,tagTagCount,tagCount,total_num_of_tokens = calculateWordTagCOunt(sentences)
    wordTagProb = calculateWordTagProb(tagCount,wordTagCount)
    tagTagProb = calculateTagTagProb(tagTagCount,tagCount)
    tagProb = calculateTagProb(tagCount,total_num_of_tokens)
    model = [wordTagCount,tagTagCount,tagCount,wordTagProb,tagTagProb,tagProb]
    print("Model Created")
    return model

def calculateTagProb(tagCount,totalNumOfTokens):
    tagProb = collections.defaultdict(float)
    for tag in tagCount.keys():
        tagProb[tag] = float(tagCount[tag])/totalNumOfTokens
    return tagProb

def calculateConfusionMatrix(sentences, predictions):
    f = open("confusion.txt","w")
    confusionMatrix = collections.defaultdict(lambda: collections.defaultdict(int))
    listOftags = []
    for index1 in range(0,len(sentences)):
        sentence = sentences[index1]
        for index2 in range(0,len(sentence)):
            xtag = sentence[index2].tag
            ytag = predictions[index1][index2].tag
            listOftags.append(xtag)
            listOftags.append(ytag)
            confusionMatrix[xtag][ytag]+=1


    ##Display the confusion matrix, asume all the tag will be present here
    tags = set(listOftags)
    print(tags)
    print("...Confusion Matrix  ..... \n")
    for tag in tags: ###RowHeadlie
        f.write(tag)
        for i in range(len(tag),5):
            f.write(" ")
    for tag1 in tags:
        f.write("\n")
        f.write(tag1)
        for i in range(len(tag),5):
            f.write(" ")
        for tag2 in tags:
            f.write(str(confusionMatrix[tag1][tag2]) )
            for i in range(len(str(confusionMatrix[tag1][tag2])), 5):
                f.write(" ")
        f.write("\n")
    f.close()

def probOfUnknown(sentences,model):
    wordTagCount, tagTagCount, tagCount, wordTagProb, tagTagProb, tagProb = model
    totalNumOfWords = 0
    numUnknownWord = 0
    for sentence in sentences:
        for word in sentence:
            totalNumOfWords+=1
            #for unknown words
            if (wordTagCount.has_key(word)==False):
                numUnknownWord+=1
    probUnknownWord = float(numUnknownWord)/totalNumOfWords
    unknownWordProb = collections.defaultdict(float)
    for tag in tagCount.keys():
        unknownWordProb[tag] = probUnknownWord*tagProb[tag]
    return unknownWordProb


def predict_tags(sentences, model):
    wordTagCount, tagTagCount, tagCount, wordTagProb, tagTagProb, tagProb = model
    #for unknwon
    unknownWordProb = probOfUnknown(sentences,model)
    print ("Total Number Of Lines {}\nProcessing .........").format(len(sentences))
    num_lines_processed = 0
    viterbi_matrix = collections.defaultdict(lambda: collections.defaultdict(float))
    #we dont need the starting of sentence tag
    all_tags = tagProb.keys()
    all_tags.remove('<s>')
    previous_winner = collections.defaultdict(lambda: collections.defaultdict(int))

    for sentence in sentences:
        # separately process firstWord first
        #the first word doesnot need three probabilities
        #prob1 = previous prob from seq
        #prob2 = probab. of word being a tag
        #prob3 = prob of tag being after a tag
        #prob 3 =
        firstWord = sentence[0].word
        for tag in all_tags:
            if not ( wordTagProb.has_key(firstWord)):
                prob1 = unknownWordProb[tag]
            else:
                prob1 = wordTagProb[firstWord][tag]
            prob2 = tagTagProb[tag]["<s>"]
            viterbi_matrix[tag][0] =prob1*prob2
        #for rest of matrix
        for column_index in range(1, len(sentence)):
            nextWord = sentence[column_index].word
            for tagForColumn in all_tags:
                temp_list = []
                for previousTag in all_tags:
                    value1 = viterbi_matrix[previousTag][column_index-1]
                    if ( wordTagProb.has_key(nextWord)):
                        value2 = wordTagProb[nextWord][tagForColumn]
                    else:
                        value2 = unknownWordProb[tagForColumn]
                    value3 = tagTagProb[tagForColumn][previousTag]
                    temp_list.append(value1*value2*value3)
                value = max(temp_list)
                index = temp_list.index(value)
                viterbi_matrix[tagForColumn][column_index ] = value
                previous_winner[tagForColumn][column_index ] = index
        # last column
        column_index = len(sentence) -1
        temp_list = []
        for each_tag in all_tags:
            temp_list.append(viterbi_matrix[each_tag][column_index])
        value = max(temp_list)
        index = temp_list.index(value)
        for column_index in range(len(sentence)-1, -1, -1):
            token = sentence[column_index]
            current_tag = all_tags[index]
            token.tag = current_tag
            index = previous_winner[current_tag][column_index]
        num_lines_processed +=1
        if((num_lines_processed % 500) == 0):
            print("Lines remaining = "+ str(len(sentences)-num_lines_processed))
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
    #calculateConfusionMatrix(test_sents,predictions)