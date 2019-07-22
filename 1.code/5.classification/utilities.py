from sklearn.utils import shuffle
import pickle
import re
import time
import torch
import numpy as np

init_time = time.time()

def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+');
    strNewRow = '\n'.join(listString);
    manipulatedData.write(strNewRow);
    manipulatedData.close();


def stringCleansing(string):
    string = string.replace("\n", "")
    string = string.replace("\"", "")
    string = string.replace("\r", "")
    string = string.strip()
    # string = string.lower()
    return string


def position_modification(pattern, string, listPosition, intAdjust):
    listMatch = []

    for m in re.finditer(pattern, string):
        listMatch.append(m)

    listMatch.reverse()
    for m in listMatch:
        if intAdjust == 1:
            string = string[0:m.start()] + " " + m.group(0) + string[m.end():]
        elif intAdjust == 2:
            string = string[0:m.start()] + " " + m.group(0) + " " + string[m.end():]
        else:
            string = string[0:m.start()] + " " + string[m.end():]
        for i in range(len(listPosition)):
            if m.start() < listPosition[i]:
                if intAdjust == 0:
                    listPosition[i] -= (len(m.group(0)) - 1)
                else:
                    listPosition[i] += intAdjust

    return string, listPosition


def clean_str(string, pos):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    re.sub: replace fuction.. reference: http://egloos.zum.com/sweeper/v/3065126
    """
    entity1Pos = pos[0]
    entity2Pos = pos[1]
    listPosition = [int(i) for i in entity1Pos.split("-")] + [int(i) for i in entity2Pos.split("-")]

    if len(pos) > 2:
        contextPos = pos[2]
        listPosition += [int(i) for i in contextPos.split("-")]

    # if len(listPosition) != 6:
    #     print(listPosition)

    # entity 1 should come before entity 2 in the sentence order
    if listPosition[0] > listPosition[2]:
        listPosition[0], listPosition[2] = listPosition[2], listPosition[0]
        listPosition[1], listPosition[3] = listPosition[3], listPosition[1]

    string = re.sub(r"[^A-Za-z0-9(),/+\-!?\'\`]", " ", string)  # punctuation -> whitespace
    listPattern = [r"\'s", r"\'ve", r"n\'t", r"\'re", r"\'d", r"\'ll"]
    for pattern in listPattern:
        string, listPattern = position_modification(pattern, string, listPosition, 1)
    # string = re.sub(r"\'s", " \'s", string)    # 's -> whitespace + 's
    # string = re.sub(r"\'ve", " \'ve", string)  # 've -> whitespace + 've
    # string = re.sub(r"n\'t", " n\'t", string)  # n't -> whitespace + n't
    # string = re.sub(r"\'re", " \'re", string)  # 're -> whitespace + 're
    # string = re.sub(r"\'d", " \'d", string)    # 'd -> whitespace + 'd
    # string = re.sub(r"\'ll", " \'ll", string)  # 'll -> whitespace + 'll

    listPattern = [r",", r"!", r"\(", r"\)", r"\)", r"\?"]
    for pattern in listPattern:
        string, listPattern = position_modification(pattern, string, listPosition, 2)

    # string = re.sub(r",", " , ", string)       # , -> whitespace + , + whitespace
    # string = re.sub(r"!", " ! ", string)       # ! -> whitespace + ! + whitespace
    # string = re.sub(r"\(", " ( ", string)      # ( -> whitespace + ( + whitespace
    # string = re.sub(r"\)", " ) ", string)      # ) -> whitespace + ) + whitespace
    # string = re.sub(r"\?", " ? ", string)     # ? -> whitespace + ? + whitespace

    listPattern = [r"\s{2,}"]
    for pattern in listPattern:
        string, listPattern = position_modification(pattern, string, listPosition, 0)

    # string = re.sub(r"\s{2,}", " ", string)    # consecutive whitespace -> single whitespace

    if string[0] == " ":
        string = string[1:]
        listPosition = [i - 1 for i in listPosition]

    return string, listPosition


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def token_index(sentence, listPosition):
    strEntity = sentence[listPosition[0]:listPosition[1]]
    strPrequel = sentence[:listPosition[0]] + "EOS"
    intEntityLength = len(strEntity.split())
    intEntityStart = len(strPrequel.split())
    return list(range(intEntityStart - 1, intEntityStart + intEntityLength - 1))


def read_context_finetuned_MGNC(boolShuffle=True):
    # True / False = 398 / 686 (1,084)
    boolPPI = False
    undersampling_rate = 1
    boolNegativeSampling = False
    boolReverse = True

    boolSaveSet = False
    intSemantic = 0

    data = {}
    dicData = {}
    setConcept = set()

    with open("../../resource/PC13GE11GE13_IDconverted.tsv", "r", encoding="utf-8") as fileInput:
    # with open("../../resource/PC13GE11GE13_sampled.tsv", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            strToken, listCharPosition = clean_str(listInstance[3], [listInstance[4], listInstance[10], listInstance[16]])
            listToken = strToken.split()

            listEntity1Index = token_index(strToken, listCharPosition[:2])
            listEntity2Index = token_index(strToken, listCharPosition[2:4])
            listContextIndex = token_index(strToken, listCharPosition[4:])

            boole1c = True if listEntity1Index[0] < listContextIndex[0] else False
            boole2c = True if listEntity2Index[0] < listContextIndex[0] else False

            # molecule entity 2 position indecator
            listToken.insert(listEntity2Index[-1] + 1, "</e2>")
            listToken.insert(listEntity2Index[0], "<e2>")

            if boole2c:
                listContextIndex = [i + 2 for i in listContextIndex]

            # molecule entity 1 position indecator
            listToken.insert(listEntity1Index[-1] + 1, "</e1>")
            listToken.insert(listEntity1Index[0], "<e1>")

            if boole1c:
                listContextIndex = [i + 2 for i in listContextIndex]

            # context entity position indecator
            listToken.insert(listContextIndex[-1] + 1, "</c>")
            listToken.insert(listContextIndex[0], "<c>")

            ####################### checking ######################
            # listEntity1Index = [i + 1 for i in listEntity1Index]
            #
            # listEntity2Index = [i + 3 for i in listEntity2Index]
            #
            # listContextIndex = [i + 1 for i in listContextIndex]
            # if not boole1c:
            #     listEntity1Index = [i + 2 for i in listEntity1Index]
            # if not boole2c:
            #     listEntity2Index = [i + 2 for i in listEntity2Index]
            #
            # print(" ".join(listToken))
            # print("\t", " ".join([listToken[i] for i in listEntity1Index]) + "\t" + " ".join([listToken[i] for i in listEntity2Index]) + "\t" + " ".join([listToken[i] for i in listContextIndex]))

            if boolPPI:
                strLabel = listInstance[24]
            else:
                # [ relation type, direction, context label, context type ]
                if listInstance[24] == "FALSE":
                    # strLabel = listInstance[24]
                    strLabel = "NA|NA"
                else:
                    strLabel = '|'.join(listInstance[22:24])

            if boolSaveSet:
                setConcept = setConcept.union(set([listInstance[7 + intSemantic], listInstance[13 + intSemantic], listInstance[19 + intSemantic]]))

            if strLabel not in dicData:
                dicData[strLabel] = [[], [], [], 0]

            # concept ID @ '|'.join(list of semantic types)
            # [c, e1, e2]

            # listTokenConcept = [listInstance[19 + intSemantic], listInstance[7 + intSemantic], listInstance[13 + intSemantic]]
            listTokenConcept = [listInstance[19] + "@" + listInstance[20], listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14]]
            # print(listInstance[0], listTokenConcept)

            # class: [ instance, concept, sentence ID, instance count in a class ]
            dicData[strLabel][0].append(listToken)
            dicData[strLabel][1].append(listTokenConcept)
            dicData[strLabel][2].append(listInstance[2])
            dicData[strLabel][3] += 1

    # print(len(dicData))
    data["train_sen"], data["train_class"], data["train_concept"], data["train_id"] = [], [], [], []
    data["dev_sen"], data["dev_class"], data["dev_concept"], data["dev_id"] = [], [], [], []
    data["test_sen"], data["test_class"], data["test_concept"], data["test_id"] = [], [], [], []

    int_second_largest = sorted([dicData[key][3] for key in dicData.keys()])[-2]

    for key in dicData.keys():
        if boolShuffle:
            dicData[key][0], dicData[key][1], dicData[key][2] = shuffle(dicData[key][0], dicData[key][1], dicData[key][2])

        if boolReverse:
            dicData[key][0].reverse()
            dicData[key][1].reverse()
            dicData[key][2].reverse()

        if boolNegativeSampling:
            if key == "NA|NA":
                dicData[key][0] = dicData[key][0][:int_second_largest*undersampling_rate]
                dicData[key][1] = dicData[key][1][:int_second_largest*undersampling_rate]

        intUnit = round(len(dicData[key][0]) / 10)
        test_idx = len(dicData[key][0]) - intUnit
        dev_idx = test_idx - intUnit

        data["train_sen"] += dicData[key][0][:dev_idx]
        data["train_concept"] += dicData[key][1][:dev_idx]
        data["train_id"] += dicData[key][2][:dev_idx]
        data["train_class"] += [key] * dev_idx

        data["dev_sen"] += dicData[key][0][dev_idx:test_idx]
        data["dev_concept"] += dicData[key][1][dev_idx:test_idx]
        data["dev_id"] += dicData[key][2][dev_idx:test_idx]
        data["dev_class"] += [key] * (test_idx - dev_idx)

        data["test_sen"] += dicData[key][0][test_idx:]
        data["test_concept"] += dicData[key][1][test_idx:]
        data["test_id"] += dicData[key][2][test_idx:]
        data["test_class"] += [key] * (len(dicData[key][0]) - test_idx)

    if boolSaveSet:
        listFileName = ["../../resource/concept_finetuned_MGNC.vocab", "../../resource/concept_semantic_finetuned_MGNC.vocab"]
        manipulatedData = open(listFileName[intSemantic], 'w+')
        strNewRow = ' '.join(list(setConcept))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concepts: ", len(setConcept))

        if intSemantic:
            concept_semantic("finetuned_MGNC")
        else:
            concept("finetuned_MGNC")

    return data


def read_context_finetuned():
    boolPPI = True
    boolSaveSet = False
    boolNegativeSampling = False
    undersampling_rate = 1

    data = {}
    setConceptComposition = set()
    setConcept = set()
    dicData = {}

    with open("../../resource/PC13GE11GE13_IDconverted.tsv", "r", encoding="utf-8") as fileInput:
    # with open("../../resource/PC13GE11GE13_sampled.tsv", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            strToken, listCharPosition = clean_str(listInstance[3], listInstance[4], listInstance[10], listInstance[16])
            listToken = strToken.split()

            boolKnowledge = False
            if not boolKnowledge:
                listEntity1Index = token_index(strToken, listCharPosition[:2])
                listEntity2Index = token_index(strToken, listCharPosition[2:4])
                listContextIndex = token_index(strToken, listCharPosition[4:])

                boole1c = True if listEntity1Index[0] < listContextIndex[0] else False
                boole2c = True if listEntity2Index[0] < listContextIndex[0] else False

                # molecule entity 2 position indecator
                listToken.insert(listEntity2Index[-1] + 1, "</e2>")
                listToken.insert(listEntity2Index[0], "<e2>")

                if boole2c:
                    listContextIndex = [i + 2 for i in listContextIndex]

                # molecule entity 1 position indecator
                listToken.insert(listEntity1Index[-1] + 1, "</e1>")
                listToken.insert(listEntity1Index[0], "<e1>")

                if boole1c:
                    listContextIndex = [i + 2 for i in listContextIndex]

                # context entity position indecator
                listToken.insert(listContextIndex[-1] + 1, "</c>")
                listToken.insert(listContextIndex[0], "<c>")

                ####################### checking ######################
                # listEntity1Index = [i + 1 for i in listEntity1Index]
                #
                # listEntity2Index = [i + 3 for i in listEntity2Index]
                #
                # listContextIndex = [i + 1 for i in listContextIndex]
                # if not boole1c:
                #     listEntity1Index = [i + 2 for i in listEntity1Index]
                # if not boole2c:
                #     listEntity2Index = [i + 2 for i in listEntity2Index]
                #
                # print(" ".join(listToken))
                # print("\t", " ".join([listToken[i] for i in listEntity1Index]) + "\t" + " ".join([listToken[i] for i in listEntity2Index]) + "\t" + " ".join([listToken[i] for i in listContextIndex]))

            if boolPPI:
                strLabel = listInstance[24]
            else:
                # [ relation type, direction, context label, context type ]
                if listInstance[24] == "FALSE":
                    strLabel = listInstance[24]
                else:
                    strLabel = '|'.join(listInstance[22:25])

            listTokenConcept = []

            # concept ID @ '|'.join(list of semantic types)
            setEntity1Index = set(token_index(strToken, listCharPosition[:2]))
            setEntity2Index = set(token_index(strToken, listCharPosition[2:4]))
            setContextIndex = set(token_index(strToken, listCharPosition[4:]))

            for i in range(len(listToken)):
                if i in setEntity1Index:
                    listTokenConcept.append(listInstance[7] + "@" + listInstance[8])
                elif i in setEntity2Index:
                    listTokenConcept.append(listInstance[13] + "@" + listInstance[14])
                elif i in setContextIndex:
                    listTokenConcept.append(listInstance[19] + "@" + listInstance[20])
                else:
                    listTokenConcept.append('-')

            if boolSaveSet:
                setConcept = setConcept.union(set([listInstance[7], listInstance[13], listInstance[19]]))
                setConceptComposition = setConceptComposition.union(
                    set([listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14],
                         listInstance[19] + "@" + listInstance[20]]))

            if strLabel not in dicData:
                dicData[strLabel] = [[], [], [], 0]

            dicData[strLabel][0].append(listToken)
            dicData[strLabel][1].append(listTokenConcept)
            dicData[strLabel][2].append(listInstance[2])
            dicData[strLabel][3] += 1

    # print(len(dicData))
    data["train_sen"], data["train_class"], data["train_concept"], data["train_id"] = [], [], [], []
    data["dev_sen"], data["dev_class"], data["dev_concept"], data["dev_id"] = [], [], [], []
    data["test_sen"], data["test_class"], data["test_concept"], data["test_id"] = [], [], [], []

    int_second_largest = sorted([dicData[key][3] for key in dicData.keys()])[-2]

    for key in dicData.keys():
        dicData[key][0], dicData[key][1], dicData[key][2] = shuffle(dicData[key][0], dicData[key][1], dicData[key][2])

        if boolNegativeSampling:
            if key == "FALSE":
                dicData[key][0] = dicData[key][0][:int_second_largest*undersampling_rate]
                dicData[key][1] = dicData[key][1][:int_second_largest*undersampling_rate]
                dicData[key][2] = dicData[key][2][:int_second_largest * undersampling_rate]

        intUnit = round(len(dicData[key][0]) / 10)
        test_idx = len(dicData[key][0]) - intUnit
        dev_idx = test_idx - intUnit

        data["train_sen"] += dicData[key][0][:dev_idx]
        data["train_class"] += [key] * dev_idx
        data["train_concept"] += dicData[key][1][:dev_idx]
        data["train_id"] += dicData[key][2][:dev_idx]

        data["dev_sen"] += dicData[key][0][dev_idx:test_idx]
        data["dev_class"] += [key] * (test_idx-dev_idx)
        data["dev_concept"] += dicData[key][1][dev_idx:test_idx]
        data["dev_id"] += dicData[key][2][dev_idx:test_idx]

        data["test_sen"] += dicData[key][0][test_idx:]
        data["test_class"] += [key] * (len(dicData[key][0])-test_idx)
        data["test_concept"] += dicData[key][1][test_idx:]
        data["test_id"] += dicData[key][2][test_idx:]

        # print(key, dev_idx, test_idx - dev_idx, len(dicData[key][0]) - test_idx)

    if boolSaveSet:
        manipulatedData = open("../../resource/concept_composition_finetuned.vocab", 'w+')
        strNewRow = ' '.join(list(setConceptComposition))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concept composition: ", len(setConceptComposition))

        manipulatedData = open("../../resource/concept_finetuned.vocab", 'w+')
        strNewRow = ' '.join(list(setConcept))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concepts: ", len(setConcept))

        concept_composition("finetuned")

    return data


def read_context_finetuned_10fold(boolShuffle=False):
    # True / False = 398 / 686 (1,084)
    boolPPI = False
    dicData = {}
    with open("../../resource/PC13GE11GE13_IDconverted.tsv", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            strToken, listCharPosition = clean_str(listInstance[3], [listInstance[4], listInstance[10], listInstance[16]])
            listToken = strToken.split()

            listEntity1Index = token_index(strToken, listCharPosition[:2])
            listEntity2Index = token_index(strToken, listCharPosition[2:4])
            listContextIndex = token_index(strToken, listCharPosition[4:])

            boole1c = True if listEntity1Index[0] < listContextIndex[0] else False
            boole2c = True if listEntity2Index[0] < listContextIndex[0] else False

            # molecule entity 2 position indecator
            listToken.insert(listEntity2Index[-1] + 1, "</e2>")
            listToken.insert(listEntity2Index[0], "<e2>")

            if boole2c:
                listContextIndex = [i + 2 for i in listContextIndex]

            # molecule entity 1 position indecator
            listToken.insert(listEntity1Index[-1] + 1, "</e1>")
            listToken.insert(listEntity1Index[0], "<e1>")

            if boole1c:
                listContextIndex = [i + 2 for i in listContextIndex]

            # context entity position indecator
            listToken.insert(listContextIndex[-1] + 1, "</c>")
            listToken.insert(listContextIndex[0], "<c>")

            if boolPPI:
                strLabel = listInstance[24]
            else:
                # [ relation type, direction, context label, context type ]
                if listInstance[24] == "FALSE":
                    # strLabel = listInstance[24]
                    strLabel = "NA|NA"
                else:
                    strLabel = '|'.join(listInstance[22:24])

            if strLabel not in dicData:
                dicData[strLabel] = [[], [], [], 0]

            intSemantic = 1
            listTokenConcept = [listInstance[19 + intSemantic], listInstance[7 + intSemantic], listInstance[13 + intSemantic]]

            # class: [ instance, concept, sentence ID, instance count in a class ]
            dicData[strLabel][0].append(listToken)
            dicData[strLabel][1].append(listTokenConcept)
            dicData[strLabel][2].append(listInstance[2])
            dicData[strLabel][3] += 1

    for key in dicData.keys():
        if boolShuffle:
            dicData[key][0], dicData[key][1], dicData[key][2] = shuffle(dicData[key][0], dicData[key][1], dicData[key][2])

    return dicData


def read_context_pretrained_MGNC(boolShuffle=True):
    boolSaveSet = False
    boolOversampling = False
    oversampling_rate = 1
    undersampling_rate = 1
    boolNegativeSampling = True
    intSemantic = 1

    data = {}
    dicData = {}
    setConceptComposition = set()

    listFile = ["../../resource/PC13GE11GE13_pretrained_IDconverted.tsv", "../../resource/EVEX_pretrained_IDconverted_sampled.tsv"]
    # listFile = ["../../resource/PC13GE11GE13_pretrained_IDconverted.tsv"]
    # listFile = ["../../resource/EVEX_pretrained_IDconverted.tsv"]

    # with open("../../resource/PC13GE11GE13_IDconverted.tsv", "r", encoding="utf-8") as fileInput:
    # with open("../../resource/PC13GE11GE13_sampled.tsv", "r", encoding="utf-8") as fileInput:
    for strFile in listFile:
        with open(strFile, "r", encoding="utf-8") as fileInput:
            for strInstance in fileInput:
                strInstance = stringCleansing(strInstance)
                listInstance = strInstance.split("\t")
                strToken, listCharPosition = clean_str(listInstance[3], [listInstance[4], listInstance[10]])
                listToken = strToken.split()

                listEntity1Index = token_index(strToken, listCharPosition[:2])
                listEntity2Index = token_index(strToken, listCharPosition[2:4])

                if listEntity1Index == listEntity2Index:
                    continue

                if len(listEntity1Index) * len(listEntity2Index) == 0:
                    # print(listInstance[4], listInstance[5], listInstance[10], listInstance[11])
                    continue

                # listContextIndex = token_index(strToken, listCharPosition[4:])
                #
                # boole1c = True if listEntity1Index[0] < listContextIndex[0] else False
                # boole2c = True if listEntity2Index[0] < listContextIndex[0] else False

                # molecule entity 2 position indecator
                listToken.insert(listEntity2Index[-1] + 1, "</e2>")
                listToken.insert(listEntity2Index[0], "<e2>")

                # if boole2c:
                #     listContextIndex = [i + 2 for i in listContextIndex]

                # molecule entity 1 position indecator
                listToken.insert(listEntity1Index[-1] + 1, "</e1>")
                listToken.insert(listEntity1Index[0], "<e1>")

                # if boole1c:
                #     listContextIndex = [i + 2 for i in listContextIndex]

                # context entity position indecator
                # listToken.insert(listContextIndex[-1] + 1, "</c>")
                # listToken.insert(listContextIndex[0], "<c>")

                ###################### checking ######################
                # listEntity1Index = [i + 1 for i in listEntity1Index]
                #
                # listEntity2Index = [i + 3 for i in listEntity2Index]
                #
                # listContextIndex = [i + 1 for i in listContextIndex]
                # if not boole1c:
                #     listEntity1Index = [i + 2 for i in listEntity1Index]
                # if not boole2c:
                #     listEntity2Index = [i + 2 for i in listEntity2Index]
                #
                # print(" ".join(listToken))
                # print(listInstance[5], listInstance[11], " ".join([listToken[i] for i in listEntity1Index]), " ".join([listToken[i] for i in listEntity2Index]))

                # [ relation type, direction ]
                strLabel = "|".join(listInstance[16:])

                # concept ID @ '|'.join(list of semantic types)
                # [c, e1, e2]
                if boolSaveSet:
                    setConceptComposition = setConceptComposition.union(set([listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14]]))

                # listTokenConcept = [listInstance[19 + intSemantic], listInstance[7 + intSemantic], listInstance[13 + intSemantic]]
                listTokenConcept = ["dummy", listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14]]
                # print(listInstance[0], listTokenConcept)

                # class: [ instance, concept, sentence ID, instance count in a class ]
                if strLabel not in dicData:
                    dicData[strLabel] = [[], [], [], 0]
                dicData[strLabel][0].append(listToken)
                dicData[strLabel][1].append(listTokenConcept)
                dicData[strLabel][2].append(listInstance[2])
                dicData[strLabel][3] += 1

    if boolNegativeSampling | boolOversampling:
        int_second_largest = sorted([dicData[key][3] for key in dicData.keys()])[-2]

    # print(len(dicData))
    data["train_sen"], data["train_class"], data["train_concept"], data["train_id"] = [], [], [], []
    data["dev_sen"], data["dev_class"], data["dev_concept"], data["dev_id"] = [], [], [], []
    data["test_sen"], data["test_class"], data["test_concept"], data["test_id"] = [], [], [], []

    for key in dicData.keys():
        if boolOversampling:
            if int_second_largest > dicData[key][3]*2:
                oversampling_rate = 1 + int(int_second_largest/dicData[key][3])

        if boolNegativeSampling:
            if key == "NA|NA":
                dicData[key][0] = dicData[key][0][:int_second_largest * undersampling_rate]
                dicData[key][1] = dicData[key][1][:int_second_largest * undersampling_rate]
                dicData[key][2] = dicData[key][2][:int_second_largest * undersampling_rate]

        if boolShuffle:
            dicData[key][0], dicData[key][1], dicData[key][2] = shuffle(dicData[key][0], dicData[key][1], dicData[key][2])

        intUnit = round(len(dicData[key][0]) / 10)
        test_idx = len(dicData[key][0]) - intUnit
        dev_idx = test_idx - intUnit

        for i in range(oversampling_rate):
            data["train_sen"] += dicData[key][0][:dev_idx]
            data["train_concept"] += dicData[key][1][:dev_idx]
            data["train_id"] += dicData[key][2][:dev_idx]
            data["train_class"] += [key] * dev_idx

        data["dev_sen"] += dicData[key][0][dev_idx:test_idx]
        data["dev_concept"] += dicData[key][1][dev_idx:test_idx]
        data["dev_id"] += dicData[key][2][dev_idx:test_idx]
        data["dev_class"] += [key] * (test_idx - dev_idx)

        data["test_sen"] += dicData[key][0][test_idx:]
        data["test_concept"] += dicData[key][1][test_idx:]
        data["test_id"] += dicData[key][2][test_idx:]
        data["test_class"] += [key] * (len(dicData[key][0]) - test_idx)

        # print(key, dev_idx, "->", len(data["train_class"]) - intLenBefore, oversampling_rate)

    # dicData = {}
    # for strLabel in data["train_class"]:
    #     if strLabel not in dicData:
    #         dicData[strLabel] = 0
    #     dicData[strLabel] += 1
    # print(dicData)

    if boolSaveSet:
        listFileName = ["../../resource/concept_pretrained_MGNC.vocab", "../../resource/concept_semantic_pretrained_MGNC.vocab"]
        manipulatedData = open(listFileName[intSemantic], 'w+')

        setEntity = set([strComposition.split("@")[intSemantic] for strComposition in setConceptComposition])
        strNewRow = ' '.join(list(setEntity))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concept compositions: ", len(setConceptComposition))
        listName = ["concept", "semantic type"]
        print(f"# of {listName[intSemantic]}: ", len(setEntity))

        if intSemantic:
            concept_semantic("pretrained_MGNC")
        else:
            concept("pretrained_MGNC")

    return data


def read_context_pretrained():
    boolSaveSet = False
    boolOversampling = False
    boolNegativeSampling = False
    undersampling_rate = 1

    data = {}
    setConceptComposition = set()
    setConcept = set()

    dicData = {}
    listFile = ["../../resource/PC13GE11GE13_pretrained_IDconverted.tsv", "../../resource/EVEX_pretrained_IDconverted.tsv"]
    for strFile in listFile:
        with open(strFile, "r", encoding="utf-8") as fileInput:
            for strInstance in fileInput:
                strInstance = stringCleansing(strInstance)
                listInstance = strInstance.split("\t")
                listToken = clean_str(listInstance[3]).split()

                # [ relation type, direction ]
                strLabel = "|".join(listInstance[16:])

                # concept ID @ '|'.join(list of semantic types)
                listTokenConcept = []
                for i in range(len(listToken)):
                    if i in token_index(listInstance[3], listInstance[4]):
                        listTokenConcept.append(listInstance[7] + "@" + listInstance[8])
                    elif i in token_index(listInstance[3], listInstance[10]):
                        listTokenConcept.append(listInstance[13] + "@" + listInstance[14])
                    else:
                        listTokenConcept.append('-')

                if boolSaveSet:
                    setConcept = setConcept.union(set([listInstance[7], listInstance[13]]))
                    setConceptComposition = setConceptComposition.union(set([listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14]]))

                if strLabel not in dicData:
                    dicData[strLabel] = [[], [], 0]

                dicData[strLabel][0].append(listToken)
                dicData[strLabel][1].append(listTokenConcept)
                dicData[strLabel][2] += 1

                # print(listInstance[2])
                # for i in range(len(listTokenConcept)):
                #     if listTokenConcept[i] != "-":
                #         print(listToken[i])

    if boolOversampling:
        int_second_largest = sorted([dicData[key][2] for key in dicData.keys()])[-2]

    data["train_sen"], data["train_class"], data["train_concept"] = [], [], []
    data["dev_sen"], data["dev_class"], data["dev_concept"] = [], [], []
    data["test_sen"], data["test_class"], data["test_concept"] = [], [], []

    for key in dicData.keys():
        oversampling_rate = 1
        if boolOversampling:
            if int_second_largest > dicData[key][2]*2:
                oversampling_rate = 1 + int(int_second_largest/dicData[key][2])

        dicData[key][0] = shuffle(dicData[key][0])
        dicData[key][1] = shuffle(dicData[key][1])

        if boolNegativeSampling:
            if key == "NA|NA":
                dicData[key][0] = dicData[key][0][:int_second_largest*undersampling_rate]
                dicData[key][1] = dicData[key][1][:int_second_largest*undersampling_rate]

        intUnit = round(len(dicData[key][0]) / 3)
        test_idx = len(dicData[key][0]) - intUnit
        dev_idx = test_idx - intUnit

        for i in range(oversampling_rate):
            data["train_sen"] += dicData[key][0][:dev_idx]
            data["train_class"] += [key] * dev_idx
            data["train_concept"] += dicData[key][1][:dev_idx]

        data["dev_sen"] += dicData[key][0][dev_idx:test_idx]
        data["dev_class"] += [key] * (test_idx - dev_idx)
        data["dev_concept"] += dicData[key][1][dev_idx:test_idx]

        data["test_sen"] += dicData[key][0][test_idx:]
        data["test_class"] += [key] * (len(dicData[key][0]) - test_idx)
        data["test_concept"] += dicData[key][1][test_idx:]

        # print(key, dev_idx, "->", len(data["train_class"]) - intLenBefore, oversampling_rate)

    dicData = {}
    for strLabel in data["train_class"]:
        if strLabel not in dicData:
            dicData[strLabel] = 0
        dicData[strLabel] += 1
    print(dicData)

    if boolSaveSet:
        manipulatedData = open("../../resource/concept_composition_pretrained.vocab", 'w+');
        strNewRow = ' '.join(list(setConceptComposition));
        manipulatedData.write(strNewRow);
        manipulatedData.close();
        print("# of concept composition: ", len(setConceptComposition))

        manipulatedData = open("../../resource/concept_pretrained.vocab", 'w+');
        strNewRow = ' '.join(list(setConcept));
        manipulatedData.write(strNewRow);
        manipulatedData.close();
        print("# of concepts: ", len(setConcept))

        concept_composition("pretrained")

    return data


def save_model(model, params):
    semantic = ""
    if params["KNOWLEDGE"] == "semantic":
        semantic = "semantic_"

    depth = ""
    if params["DEPTH"]:
        depth = "_depth"

    filter_size = ",".join([str(size) for size in params['FILTERS']])

    path = f"../../result/saved_models{depth}(fn{params['FILTER_NUM_CONCEPT']})/{params['DATASET']}_{semantic}{params['MODALITY']}_0.001_{filter_size}.pt"
    # pickle.dump(model, open(path, "wb"))
    # print(f"A result is saved successfully as {path}!")
    torch.save(model.module.state_dict(), path)
    # print(f"Model in {path} saved successfully!")


def save_dictionary(dic, key):
    path = f"../../result/saved_dictionary/context_{key}.tsv"
    writeOutput([str(strKey) + "\t" + str(dic[strKey]) for strKey in dic.keys()], path)


def load_dictionary(key):
    dic = {}
    path = f"../../result/saved_dictionary/context_{key}.tsv"
    with open(path, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringCleansing(strInstance).split("\t")
            if "idx_to_" in key:
                dic[int(listInstance[0])] = listInstance[1]
            else:
                dic[listInstance[0]] = int(listInstance[1])
    return dic


def save_class(list, key):
    path = f"../../result/saved_dictionary/context_class_{key}.tsv"
    writeOutput([str(i) + "\t" + list[i] for i in range(len(list))], path)


def load_model(model, params):
    semantic = ""
    if params["KNOWLEDGE"] == "semantic":
        semantic = "semantic_"

    filter_size = ",".join([str(size) for size in params['FILTERS']])

    dataset = "context_pretrained_MGNC"
    if "unlabeled" in {params['DATASET']}:
        dataset = "context_finetuned_MGNC"

    depth = ""
    if params["DEPTH"]:
        depth = "_depth"

    # path = f"../../result/saved_models(fn{params['FILTER_NUM_CONCEPT']})/{dataset}_{semantic}{params['MODALITY']}_0.001_{filter_size}.pt"
    path = f"../../result/saved_models{depth}(fn{params['FILTER_NUM_CONCEPT']})/{dataset}_{semantic}{params['MODALITY']}_0.001_{filter_size}.pt"
    # try:
    #     model = pickle.load(open(path, "rb"))
    #     # print(model)
    #     # print(f"Model in {path} loaded successfully!")
    #     # print(model)
    #     return model
    # except:
    #     print(f"No available model such as {path}.")
    #     exit()

    pretrained_weights = torch.load(path)
    state = model.state_dict()
    if params["MODALITY"] in set(["N", "ND", "NK", "NDK"]):
        npEmbedding1 = np.load("../../result/saved_dictionary/ookb_n_finetuned.npy")
        npEmbedding1 = np.concatenate([npEmbedding1, [np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16")]], axis=0)
        npEmbedding1 = np.concatenate([npEmbedding1, [np.zeros(params["WORD_DIM"]).astype("float16")]], axis=0)
        state['embedding1.weight'] = torch.cat((pretrained_weights['embedding1.weight'][:-2], torch.from_numpy(npEmbedding1).to('cuda')), 0)
        for i in range(len(params["FILTERS"])):
            state[f'conv_n_{i}.weight'] = pretrained_weights[f'conv_n_{i}.weight']
            state[f'conv_n_{i}.bias'] = pretrained_weights[f'conv_n_{i}.bias']
            state[f'bn_n_{i}.weight'] = pretrained_weights[f'bn_n_{i}.weight']
            state[f'bn_n_{i}.bias'] = pretrained_weights[f'bn_n_{i}.bias']
            state[f'bn_n_{i}.running_mean'] = pretrained_weights[f'bn_n_{i}.running_mean']
            state[f'bn_n_{i}.running_var'] = pretrained_weights[f'bn_n_{i}.running_var']
            state[f'bn_n_{i}.num_batches_tracked'] = pretrained_weights[f'bn_n_{i}.num_batches_tracked']

    if params["MODALITY"] in set(["D", "ND", "DK", "NDK"]):
        npEmbedding2 = np.load("../../result/saved_dictionary/ookb_d_finetuned.npy")
        npEmbedding2 = np.concatenate([npEmbedding2, [np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16")]], axis=0)
        npEmbedding2 = np.concatenate([npEmbedding2, [np.zeros(params["WORD_DIM"]).astype("float16")]], axis=0)
        state['embedding2.weight'] = torch.cat((pretrained_weights['embedding2.weight'][:-2], torch.from_numpy(npEmbedding2).float().to('cuda')), 0)
        for i in range(len(params["FILTERS"])):
            state[f'conv_d_{i}.weight'] = pretrained_weights[f'conv_d_{i}.weight']
            state[f'conv_d_{i}.bias'] = pretrained_weights[f'conv_d_{i}.bias']
            state[f'bn_d_{i}.weight'] = pretrained_weights[f'bn_d_{i}.weight']
            state[f'bn_d_{i}.bias'] = pretrained_weights[f'bn_d_{i}.bias']
            state[f'bn_d_{i}.running_mean'] = pretrained_weights[f'bn_d_{i}.running_mean']
            state[f'bn_d_{i}.running_var'] = pretrained_weights[f'bn_d_{i}.running_var']
            state[f'bn_d_{i}.num_batches_tracked'] = pretrained_weights[f'bn_d_{i}.num_batches_tracked']

    if params["MODALITY"] in set(["NK", "DK", "NDK"]):
        npEmbedding3 = np.load("../../result/saved_dictionary/ookb_k_finetuned.npy")
        npEmbedding3 = np.concatenate([npEmbedding3, [np.zeros(params["CONCEPT_DIM"]).astype("float16")]], axis=0)
        state['embedding3.weight'] = torch.cat((pretrained_weights['embedding3.weight'][:-1], torch.from_numpy(npEmbedding3).to('cuda')), 0)
        # print(pretrained_weights['embedding3.weight'])
        state[f'conv_k_0.weight'] = pretrained_weights[f'conv_k_0.weight']
        state[f'conv_k_0.bias'] = pretrained_weights[f'conv_k_0.bias']
        state[f'bn_k_0.weight'] = pretrained_weights[f'bn_k_0.weight']
        state[f'bn_k_0.bias'] = pretrained_weights[f'bn_k_0.bias']
        state[f'bn_k_0.running_mean'] = pretrained_weights[f'bn_k_0.running_mean']
        state[f'bn_k_0.running_var'] = pretrained_weights[f'bn_k_0.running_var']
        state[f'bn_k_0.num_batches_tracked'] = pretrained_weights[f'bn_k_0.num_batches_tracked']

    model.load_state_dict(state)
    return model


def npy_load(strModel):
    # strModel = "../../word2vec/result/word2vecf_test"
    # vec = torch.from_numpy(np.load(strModel + ".npy"))
    vec = np.load(strModel + ".npy")

    vocab = {}
    intIndex = 0
    with open(strModel + ".vocab") as f:
        for strVocab in f.read().split():
            vocab[strVocab] = intIndex
            intIndex += 1

    return vocab, vec


def concept_composition(dataset):
    with open("../../resource/concept_composition_" + dataset + ".vocab", "r", encoding="utf-8") as fileInput:
        listConceptComposition = fileInput.readline().split()

    # concept_vocab, concept_vectors = npy_load(
    #     "../../../UMLS2vec/result/concept_partial/checkpoints/concept_partial")
    concept_vocab, concept_vectors = npy_load(
        "../../../UMLS2vec/result/concept_source_or/checkpoints/concept_source_or")
    semantic_vocab, semantic_vectors = npy_load("../../../UMLS2vec/result/semantic/checkpoints/semantic")
    composition_matrix = []

    # setUnkown = set()
    for strConceptComposition in listConceptComposition:
        listInstance = strConceptComposition.split("@")
        # print(listInstance)
        if listInstance[0] in concept_vocab:
            composition_matrix.append(np.concatenate((concept_vectors[concept_vocab[listInstance[0]]], np.mean(
                [semantic_vectors[semantic_vocab[strType]] for strType in listInstance[1].split('|')], axis=0)),
                                                     axis=0))
            # print(listInstance[0])
            # setUnkown.add(listInstance[0])

        else:
            if listInstance[1] == "NA":
                composition_matrix.append(np.concatenate((np.random.uniform(-0.01, 0.01, 190).astype("float16"),
                                                         np.random.uniform(-0.01, 0.01, 10).astype("float16")), axis=0))
            else:
                composition_matrix.append(np.concatenate((np.random.uniform(-0.01, 0.01, 190).astype("float16"),
                                                          np.mean(
                                                              [semantic_vectors[semantic_vocab[strType]] for strType in
                                                               listInstance[1].split('|')], axis=0)), axis=0))
                print("OOKB: ", listInstance[0])

    np.save("../../resource/concept_composition_" + dataset + ".npy", composition_matrix)
    # print(len(setUnkown))


def concept_semantic(dataset):
    with open("../../resource/concept_semantic_" + dataset + ".vocab", "r", encoding="utf-8") as fileInput:
        listConcept = fileInput.readline().split()

    semantic_vocab, semantic_vectors = npy_load("../../../UMLS2vec/result/semantic/checkpoints/semantic")
    concept_matrix = []

    for strConcept in listConcept:
        if "@" in strConcept:
            strSemantic = strConcept.split("@")[1]
            if strSemantic in semantic_vocab:
                concept_matrix.append(np.mean([semantic_vectors[semantic_vocab[strType]] for strType in strSemantic.split('|')], axis=0))
            else:
                concept_matrix.append(np.random.uniform(-0.01, 0.01, 10).astype("float16"))
        else:
            concept_matrix.append(np.zeros(10).astype("float16"))

    np.save("../../resource/concept_semantic_" + dataset + ".npy", concept_matrix)


def concept(dataset):
    with open("../../resource/concept_" + dataset + ".vocab", "r", encoding="utf-8") as fileInput:
        listConcept = fileInput.readline().split()
    concept_vocab, concept_vectors = npy_load(
        "../../../UMLS2vec/result/concept_source_or/checkpoints/concept_source_or")
    concept_matrix = []

    for strConcept in listConcept:
        strConcept = strConcept.split("@")[0]
        if strConcept in concept_vocab:
            concept_matrix.append(np.mean([concept_vectors[concept_vocab[strConcept]]], axis=0))
        elif strConcept == "dummy":
            concept_matrix.append(np.zeros(190).astype("float16"))
        else:
            concept_matrix.append(np.random.uniform(-0.01, 0.01, 190).astype("float16"))
            print("OOKB: ", strConcept)

    np.save("../../resource/concept_" + dataset + ".npy", concept_matrix)


def distance_measure():
    boolPPI = True
    dic_e1_e2_distance = {"TOTAL": {}}
    dic_e1_c_distance = {"TOTAL": {}}
    dic_e2_c_distance = {"TOTAL": {}}
    dic_max_distance = {"TOTAL": {}}
    whole_max_distance = 0

    with open("../../resource/PC13GE11GE13_IDconverted.tsv", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            strToken, listCharPosition = clean_str(listInstance[3], listInstance[4], listInstance[10], listInstance[16])
            # listToken = strToken.split()

            if boolPPI:
                strLabel = listInstance[24]
            else:
                # [ relation type, direction, context label, context type ]
                if listInstance[24] == "FALSE":
                    strLabel = listInstance[24]
                else:
                    strLabel = '|'.join(listInstance[22:25])

            if strLabel not in dic_e1_e2_distance:
                dic_e1_e2_distance[strLabel] = {}
            if strLabel not in dic_e1_c_distance:
                dic_e1_c_distance[strLabel] = {}
            if strLabel not in dic_e2_c_distance:
                dic_e2_c_distance[strLabel] = {}
            if strLabel not in dic_max_distance:
                dic_max_distance[strLabel] = {}

            listEntity1Index = token_index(strToken, listCharPosition[:2])
            listEntity2Index = token_index(strToken, listCharPosition[2:4])
            listContextIndex = token_index(strToken, listCharPosition[4:])

            e1_e2_distance = min(abs(min(listEntity1Index) - max(listEntity2Index)), abs(min(listEntity2Index) - max(listEntity1Index)))
            e1_c_distance = min(abs(min(listEntity1Index) - max(listContextIndex)), abs(min(listContextIndex) - max(listEntity1Index)))
            e2_c_distance = min(abs(min(listEntity2Index) - max(listContextIndex)), abs(min(listContextIndex) - max(listEntity2Index)))

            temp_max_distance = max(e1_e2_distance, e1_c_distance, e2_c_distance)
            whole_max_distance = max(temp_max_distance, whole_max_distance)

            if e1_e2_distance not in dic_e1_e2_distance[strLabel]:
                dic_e1_e2_distance[strLabel][e1_e2_distance] = 1
            else:
                dic_e1_e2_distance[strLabel][e1_e2_distance] += 1
            if e1_e2_distance not in dic_e1_e2_distance["TOTAL"]:
                dic_e1_e2_distance["TOTAL"][e1_e2_distance] = 1
            else:
                dic_e1_e2_distance["TOTAL"][e1_e2_distance] += 1

            if e1_c_distance not in dic_e1_c_distance[strLabel]:
                dic_e1_c_distance[strLabel][e1_c_distance] = 1
            else:
                dic_e1_c_distance[strLabel][e1_c_distance] += 1
            if e1_c_distance not in dic_e1_c_distance["TOTAL"]:
                dic_e1_c_distance["TOTAL"][e1_c_distance] = 1
            else:
                dic_e1_c_distance["TOTAL"][e1_c_distance] += 1

            if e2_c_distance not in dic_e2_c_distance[strLabel]:
                dic_e2_c_distance[strLabel][e2_c_distance] = 1
            else:
                dic_e2_c_distance[strLabel][e2_c_distance] += 1
            if e2_c_distance not in dic_e2_c_distance["TOTAL"]:
                dic_e2_c_distance["TOTAL"][e2_c_distance] = 1
            else:
                dic_e2_c_distance["TOTAL"][e2_c_distance] += 1

            if temp_max_distance not in dic_max_distance[strLabel]:
                dic_max_distance[strLabel][temp_max_distance] = 1
            else:
                dic_max_distance[strLabel][temp_max_distance] += 1
            if temp_max_distance not in dic_max_distance["TOTAL"]:
                dic_max_distance["TOTAL"][temp_max_distance] = 1
            else:
                dic_max_distance["TOTAL"][temp_max_distance] += 1

    listWrite = []
    # listLine = [str(i + 1) for i in range(max_distance)]
    # listLine.insert(0, "label/distance")
    # listWrite.append("\t".join(listLine))
    for key in dic_e1_e2_distance.keys():
        listLine = [0] * whole_max_distance
        for distance in dic_e1_e2_distance[key].keys():
            listLine[distance - 1] = dic_e1_e2_distance[key][distance]
        listLine.insert(0, "e1_e2_" + key)
        listWrite.append("\t".join([str(i) for i in listLine]))

    for key in dic_e1_c_distance.keys():
        listLine = [0] * whole_max_distance
        for distance in dic_e1_c_distance[key].keys():
            listLine[distance - 1] = dic_e1_c_distance[key][distance]
        listLine.insert(0, "e1_c_" + key)
        listWrite.append("\t".join([str(i) for i in listLine]))

    for key in dic_e2_c_distance.keys():
        listLine = [0] * whole_max_distance
        for distance in dic_e2_c_distance[key].keys():
            listLine[distance - 1] = dic_e2_c_distance[key][distance]
        listLine.insert(0, "e2_c_" + key)
        listWrite.append("\t".join([str(i) for i in listLine]))

    for key in dic_max_distance.keys():
        listLine = [0] * whole_max_distance
        for distance in dic_max_distance[key].keys():
            listLine[distance - 1] = dic_max_distance[key][distance]
        listLine.insert(0, "max_" + key)
        listWrite.append("\t".join([str(i) for i in listLine]))

    writeOutput(listWrite, "../../resource/distance_measure.tsv")


def ookb():
    from gensim.models.keyedvectors import KeyedVectors

    data = read_context_finetuned_MGNC(False)

    word_to_idx = load_dictionary("word_to_idx")
    idx_to_word = load_dictionary("idx_to_word")
    concept_to_idx = load_dictionary("concept_to_idx")
    idx_to_concept = load_dictionary("idx_to_concept")

    loaded_vocab = set(word_to_idx.keys())
    idx = len(loaded_vocab)
    word_vectors = KeyedVectors.load_word2vec_format("../../../word2vec/result/word2vec_whole.model.bin", binary=True)
    wv_matrix = []
    word_dep_vocab, word_dep_vectors = npy_load("../../../word2vecf/result/word2vecf_200_min5_np_2015")
    wvf_matrix = []
    new_vocab = sorted(list(set([w for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"] for w in sent])))

    for vocab in new_vocab:
        if vocab not in loaded_vocab:
            boolN = True if vocab in word_vectors.vocab else False
            boolD = True if vocab in word_dep_vocab else False

            if boolN or boolD:
                word_to_idx[vocab] = idx
                idx_to_word[idx] = vocab
                idx += 1
                if boolN:
                    wv_matrix.append(word_vectors.word_vec(vocab))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, 200).astype("float16"))

                if boolD:
                    wvf_matrix.append(word_dep_vectors[word_dep_vocab[vocab]])
                else:
                    wvf_matrix.append(np.random.uniform(-0.01, 0.01, 200).astype("float16"))

    loaded_concept = set(concept_to_idx.keys())
    idx = len(loaded_concept)
    semantic_vocab, semantic_vectors = npy_load("../../../UMLS2vec/result/semantic/checkpoints/semantic")
    concept_matrix = []

    new_concept = sorted(list(set([w.split("@")[1] for sent in data["train_concept"] + data["dev_concept"] + data["test_concept"] for w in sent])))
    for con in new_concept:
        if con not in loaded_concept:
            concept_to_idx[con] = idx
            idx_to_concept[idx] = con
            idx += 1
            vector = np.mean([semantic_vectors[semantic_vocab[sType]] for sType in con.split("|")], axis=0)
            concept_matrix.append(vector)

    save_dictionary(word_to_idx, "word_to_idx_finetuned")
    save_dictionary(idx_to_word, "idx_to_word_finetuned")
    save_dictionary(concept_to_idx, "concept_to_idx_finetuned")
    save_dictionary(idx_to_concept, "idx_to_concept_finetuned")

    np.save("../../result/saved_dictionary/ookb_n_finetuned.npy", wv_matrix)
    np.save("../../result/saved_dictionary/ookb_d_finetuned.npy", wvf_matrix)
    np.save("../../result/saved_dictionary/ookb_k_finetuned.npy", concept_matrix)

    print(np.array(wv_matrix).shape)
    print(np.array(wvf_matrix).shape)
    print(np.array(concept_matrix).shape)


def model_check():
    import glob
    listModel = glob.glob("../../result/saved_models/*.pt")
    for model in listModel:
        pretrained_weights = torch.load(model)
        print(model)
        print(pretrained_weights.keys())


if __name__ == '__main__':
    # read_context_pretrained()
    # read_context_finetuned()
    # concept_composition()
    # read_context_PI()
    # distance_measure()
    # read_context_finetuned_MGNC()
    # read_context_pretrained_MGNC()
    # load_model("", "")
    # ookb()
    model_check()
    measure()

