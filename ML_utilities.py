import re
import string
from random import shuffle
import math
spamMessages = 0.0
hamMessages = 0.0
completeDict = {}
completeDict['spam'] = {}
completeDict['ham'] = {}
testDataList = []   #Has whole cleaned messages as list contents. 
trainingDataList = []

def getCount(docTextList):
    global spamMessages
    global hamMessages
    for message in docTextList:
        if 'spam' in message:
            spamMessages += 1
        else:
            hamMessages += 1
    print "hamMessages: ",hamMessages
    
def getCountPersonal():
    hamCount = 0
    for message in testDataList:
        if 'ham' in message:
            hamCount += 1
    print "Ham Messages in test data: ",hamCount
    print "Total test data: ",len(testDataList)
def splitDataSet(spamFileName, trainingPercentage=0):
    global trainingDataList
    global testDataList
    doc = open(spamFileName, 'r')
    doc = doc.read().lower()
    docTextList = doc.split("\n")
    getCount(docTextList)
    shuffle(docTextList)
    trainingDataNumber = int((len(docTextList)*trainingPercentage)/100)
    for i in range(0,trainingDataNumber+1):
        if docTextList[i] != '':
            trainingDataList.append(cleanData(docTextList[i]))      #Splitting training and test data set, storing cleaned data

    for i in range(trainingDataNumber+1, len(docTextList)):
        if docTextList[i] != '':
            testDataList.append(cleanData(docTextList[i]))

    getCountPersonal()

def cleanData2(line):
    line = line.replace("\t", " ")    #Removing tabs
#    print line
    line = re.sub("[,\/\\!\:;-_'\"\.$#@&*\\?]+"," ",line)
#    print line
    return line
    
def cleanData(line):
    allowedChars = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890 # *')
    line = line.replace("\t", " ")    #Removing tabs
    line =(''.join(filter(allowedChars.__contains__, line)))     #Removing punctuations
    return line

def getOccurenceCount():
    for line in trainingDataList:
        if line == '':
            continue
        frequencyDict = {}
        line = line.split()
        spamOrHam = line[0].lower()   #spam/ham
        line.pop(0)    #['I', 'am', 'telling', 'you']
        for word in line:
            frequencyDict[word] = getFrequency(word, line)

        addToOverallDict(spamOrHam, frequencyDict)

def addToOverallDict(spamOrHam, frequencyDict):
    global countDict
    for key, value in frequencyDict.items():
        if key in completeDict[spamOrHam].keys():
            completeDict[spamOrHam][key]['count'] += 1
        else:
            completeDict[spamOrHam][key] = {}
            completeDict[spamOrHam][key]['count'] = value
            completeDict[spamOrHam][key]['probability'] = 0.0
    

def mainResult(alpha, N):
    truePositive = 0.0
    trueNegative = 0.0
    falsePositive = 0.0
    falseNegative = 0.0
    
    for line in testDataList:
        learnedResult = classify(line, alpha, N)

        if learnedResult=='spam' and 'spam' in line:
            truePositive +=1
        elif learnedResult=='spam' and 'ham' in line:
            falsePositive +=1
        elif learnedResult=='ham' and 'ham' in line:
            trueNegative +=1
        elif learnedResult=='ham' and 'spam' in line:
            falseNegative +=1

    print "STATS WITH ALPHA= ",alpha
    print "TRUE POSITIVE: ",truePositive
    print "TRUE NEGATIVE: ",trueNegative
    print "FALSE POSITIVE: ",falsePositive
    print "FALSE NEGATIVE: ",falseNegative
    precision = float(truePositive)/float(truePositive+falsePositive)*100
    recall = float(truePositive)/float(truePositive+falseNegative)*100
    fscore = float(2*precision*recall)/float(precision+recall)
    accuracy = float(truePositive+trueNegative)*100/float(truePositive+trueNegative+falsePositive+falseNegative)

    print "\n"
    print "PRECISION: ",precision
    print "RECALL: ",recall
    print "FSCORE: ",fscore
    print "ACCURACY: ",accuracy

def classify(line, alpha, N):
    print completeDict
    wordList = line.split()
#    print "LABEL ",wordList[0]
    wordList = wordList.pop(0)
    pGivenSpam = 1.0
    pGivenHam = 1.0
    for word in wordList:
        if word in completeDict['spam'].keys():
#            print "SPAM: {}  {}".format(word, completeDict['spam'][word]['probability'])
            pGivenSpam *= completeDict['spam'][word]['probability']
        else:
#            print "SPAM: {}  {}".format(word, float(getSum('spam')+(alpha*N)))
            pGivenSpam *= alpha/float(getSum('spam')+(alpha*N))            

        if word in completeDict['ham'].keys():
#            print "HAM: {}  {}".format(word, completeDict['ham'][word]['probability'])
            pGivenHam *= completeDict['ham'][word]['probability']
        else:
#            print "HAM: {}  {}".format(word, float(getSum('ham')+(alpha*N)))
            pGivenHam *= alpha/float(getSum('ham')+(alpha*N))

#    print "Probabilities: ", pGivenSpam,pGivenHam
    pGivenSpam *= (spamMessages/(spamMessages+hamMessages))
    pGivenHam *= (hamMessages/(spamMessages+hamMessages))
    if pGivenSpam > pGivenHam:
        return 'spam'
    else:
        return 'ham'
def getSum(identifier):
    count = 0
    for word in completeDict[identifier].keys():
        count += completeDict[identifier][word]['count']

    return count
    
def getFrequency(word, wordList):
    count = 0
    for wordEntry in wordList:
        if word == wordEntry:
            count += 1

    return count

def getProbabilityPerWord(alpha, N):
    for key in completeDict.keys():
        for word in completeDict[key].keys():
           completeDict[key][word]['probability']  = (completeDict[key][word]['count'] + alpha)/(getSum(key)+(alpha*N))


def modelWrapper(fileName, trainingPercentage, alpha, N):
    splitDataSet(fileName, trainingPercentage=trainingPercentage)
    getOccurenceCount()
    getProbabilityPerWord(alpha, N)
    mainResult(alpha, N)

modelWrapper('C:\\Users\\varsh\\Sem 1\\ML\\HW 1\\SMSSpamCollection.txt', trainingPercentage=80, alpha=0.5, N=20000)

alphaList = [-5,-4,-3,-2,-1,0]
#splitDataSet(fileName, trainingPercentage=trainingPercentage)
#getOccurenceCount()
#for alpha in alphaList:
#    modelWrapper('C:\\Users\\varsh\\Sem 1\\ML\\HW 1\\SMSSpamCollection.txt', trainingPercentage=80, alpha=2**alpha, N=20000)
#    getProbabilityPerWord(alpha=2**alpha, N=20000)
#    mainResult(alpha=2**alpha, N=20000)
