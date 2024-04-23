#!/usr/bin/python3
from urllib.parse import parse_qs
import os
import random
import nltk
from nltk.corpus import wordnet as wn

def application(env, start_response):
    os.environ["NLTK_DATA"] = "/kunden/homepages/6/d227567708/htdocs/nltk_data"
    start_response('200 OK', [('Content-Type','text/html')])
    try:
        request_body_size = int(env.get('CONTENT_LENGTH', 0))
    except (ValueError):
        request_body_size = 0
    request_body = env['wsgi.input'].read(request_body_size)
    params = parse_qs(request_body)
    keys = list(params)
    firstKey = keys[0]
    firstArg = params[firstKey][0]
    finalArg = generateMeaningSample(25, str(firstArg))
    return [b"%s" % finalArg.encode()]

def generateMeaningSample(sentenceLength, inputString):
	wordList = createWordList(inputString)
	meaningList = createMeaningList(wordList)
	markovList = generateMarkovChain(wordList)
	meaningChain = generateMeaningChain(meaningList)
	finalSample = "start"
	while finalSample[-1] != "." and finalSample[-1] != "?" and finalSample[-1] != "!":
		finalSample = startMeaningTextFragmentGeneration(markovList, meaningChain, sentenceLength)
	return finalSample

def createWordList(inputString):
	wordList = inputString.split()
	return wordList

def createMeaningList(wordList):
	meaningList = []
	for tmp in wordList:
		word = cleanUpPunctuationOf(tmp)[0]
		if wn.synsets(word) != []:
			meaningList.append(random.choice(wn.synsets(word)))
		else:
			meaningList.append(word)
	return meaningList

def generateMarkovChain(wordList):
	countVar = 0
	markovList = []

	for x in wordList: 
		if countVar + 1 != len(wordList):
			markovList.append(wordList[countVar] + " " + wordList[countVar + 1])
		countVar += 1
	return markovList

def generateMeaningChain(meaningList):
	countVar = 0
	meaningChain = []

	for x in meaningList: 
		if countVar + 1 != len(meaningList):
			meaningChain.append(meaningList[countVar])
		countVar += 1
	return meaningChain

def startMeaningTextFragmentGeneration(markovList, meaningChain, sentenceLength):
	if len(markovList) - sentenceLength > 0:
		sentence = ""
		startNum = 0
		while True:
			startNum = random.randint(0, len(markovList) - sentenceLength)
			if markovList[startNum].split()[0][0].isupper():
				break
		return createMeaningTextFragment(sentenceLength, markovList, meaningChain, startNum)
	else:
		return "Please input a longer text sample or generate a shorter sample."

def cleanUpPunctuationOf(word):
	punct = ""
	if word[0] == '"':
		word = word[1:len(word)]
	if word[-1] == '"':
		word = word[0:len(word) - 1]
	if (word[-1] == "." or
		word[-1] == "," or
		word[-1] == ":" or
		word[-1] == ";" or
		word[-1] == "?" or
		word[-1] == "!" or
		word[-1] == "-") and len(word) !=2:
		punct = word[-1]
		word = word[0:len(word) - 1]
	return [word, punct]

def createMeaningTextFragment(sentenceLength, markovList, meaningChain, startNum):

	combinedChain = combineChains(markovList, meaningChain)
	sentence = combinedChain[startNum][0].split()[0]
	curWord = combinedChain[startNum][0].split()[1]
	previousMeaning = generateTrueMeanings(meaningChain)[0]

	for wordCount in range(0, sentenceLength):
		
		nextWordCandidates = findCandidates(combinedChain, curWord)
		if nextWordCandidates == []: # we're done
			break
		#print(nextWordCandidates)

		sentence += " " + curWord
		#print(sentence)

		meaningIndexOrWord = chooseNextWord(nextWordCandidates, previousMeaning)
		#print("meaningIndexOrWord: ", meaningIndexOrWord)

		if type(meaningIndexOrWord) is str:
			curWord = meaningIndexOrWord
		else:
			curWord = nextWordCandidates[meaningIndexOrWord][0].split()[1]
			previousMeaning = nextWordCandidates[meaningIndexOrWord][1]

	return sentence

def combineChains(markovList, meaningChain):
	combinedChain = []
	counter = 0
	for x in range(0, len(markovList)):
		tempTuple = (markovList[counter], meaningChain[counter])
		combinedChain.append(tempTuple)
		counter += 1
	return combinedChain

def generateTrueMeanings(meaningChain):
	trueChain = []
	for x in meaningChain:
		if type(x) is nltk.corpus.reader.wordnet.Synset:
			trueChain.append(x)
	return trueChain

def findCandidates(combinedChain, nextWord):
	nextWordCandidates = []
	for y in combinedChain:
			if y[0].split()[0] == nextWord:
				nextWordCandidates.append(y)
	return nextWordCandidates

def chooseNextWord(nextWordCandidates, previousMeaning):
	closeness = 2.0
	closestIndex = 0
	curIndex = 0
	nonSynsetList = []
	isOpenClass = False
	for candidate in nextWordCandidates:
		if type(candidate[1]) is not nltk.corpus.reader.wordnet.Synset:
			#print("candidate[1] is: ", candidate[1])
			#print("candidate[0].split()[1]: ", candidate[0].split()[1])
			nonSynsetList.append(candidate[0].split()[1])
		else:
			if previousMeaning.path_similarity(candidate[1]) == None:
				curIndex += 1
				continue
			elif previousMeaning.path_similarity(candidate[1]) < closeness:

				closeness = previousMeaning.path_similarity(candidate[1])
				closestIndex = curIndex
			isOpenClass = True
		curIndex += 1
		
	if nonSynsetList == []:
		return closestIndex
	elif isOpenClass is False:
		return random.choice(nonSynsetList)
	else:
		if random.randint(0, 1) == 0:
			return random.choice(nonSynsetList)
		else:
			return closestIndex
