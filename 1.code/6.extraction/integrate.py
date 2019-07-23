import time
import glob
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
    manipulatedData = open(strOutputName, 'w+')
    strNewRow = '\n'.join(listString)
    manipulatedData.write(strNewRow)
    manipulatedData.close()


def stringCleansing(string):
    string = string.replace("\n", "")
    string = string.replace("\"", "")
    string = string.replace("\r", "")
    string = string.strip()
    return string


def frequency_of_frequency():
    dicCount = {}
    for strResult in glob.glob("../result/raw_result/unlabeled_*.tsv"):
        with open(strResult, "r", encoding="utf-8") as fileInput:
            for strInstance in fileInput:
                strInstance = stringCleansing(strInstance)
                listInstance = strInstance.split("\t")
                if len(listInstance) > 20:
                    if listInstance[7] == listInstance[13]:
                        continue
                    if listInstance[21] == "False":
                        continue

                    key_triplet = "_".join(sorted([listInstance[7], listInstance[13]])) + "@" + listInstance[19]
                    key_relation2 = ""
                    if "Forward" in listInstance[21]:
                        key_relation = "\t".join([listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                    elif "Backward" in listInstance[21]:
                        key_relation = "\t".join([listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])
                    elif "NA" in listInstance[21]:
                        key_relation = "\t".join([listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                        key_relation2 = "\t".join([listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])

                    if key_triplet in dicCount:
                        if key_relation in dicCount[key_triplet]:
                            dicCount[key_triplet][key_relation] += 1
                        else:
                            dicCount[key_triplet][key_relation] = 1

                        if key_relation2 != "":
                            if key_relation2 in dicCount[key_triplet]:
                                dicCount[key_triplet][key_relation2] += 1
                            else:
                                dicCount[key_triplet][key_relation2] = 1
                    else:
                        if key_relation2 == "":
                            dicCount[key_triplet] = {key_relation: 1}
                        else:
                            dicCount[key_triplet] = {key_relation: 1, key_relation2: 1}

    dicCountCount = {}
    for key_triplet in dicCount.keys():
        for intFrequency in dicCount[key_triplet].values():
            if intFrequency in dicCountCount:
                dicCountCount[intFrequency] += 1
            else:
                dicCountCount[intFrequency] = 1

    intTotal = sum(dicCountCount.values())
    intSignificant = intTotal*0.05
    # print(intTotal)
    sorted_x = sorted(dicCountCount.items(), key=lambda kv: kv[0])
    listWrite = []
    boolSwitch = True
    threshold = 0
    for tup in sorted_x:
        listWrite.append(str(tup[0]) + "\t" + str(tup[1]) + "\t" + str(intTotal))
        if boolSwitch & (intTotal <= intSignificant):
            threshold = tup[0]
            boolSwitch = False
        intTotal = intTotal - tup[1]

    writeOutput(listWrite, "../result/integrated/frequency_of_frequency.tsv")
    return threshold


def conflict_voting():
    threshold = frequency_of_frequency()
    error_range = threshold
    print(threshold)
    dicCount = {}
    for strResult in glob.glob("../result/raw_result/unlabeled_*.tsv"):
        with open(strResult, "r", encoding="utf-8") as fileInput:
            for strInstance in fileInput:
                strInstance = stringCleansing(strInstance)
                listInstance = strInstance.split("\t")
                if len(listInstance) > 20:
                    if listInstance[7] == listInstance[13]:
                        continue
                    if listInstance[21] == "False":
                        continue

                    key_triplet = "_".join(sorted([listInstance[7], listInstance[13]])) + "@" + listInstance[19]
                    key_relation2 = ""
                    if "Forward" in listInstance[21]:
                        key_relation = "\t".join([listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                    elif "Backward" in listInstance[21]:
                        key_relation = "\t".join([listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])
                    elif "NA" in listInstance[21]:
                        key_relation = "\t".join([listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                        key_relation2 = "\t".join([listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])

                    if key_triplet in dicCount:
                        if key_relation in dicCount[key_triplet]:
                            dicCount[key_triplet][key_relation] += 1
                        else:
                            dicCount[key_triplet][key_relation] = 1

                        if key_relation2 != "":
                            if key_relation2 in dicCount[key_triplet]:
                                dicCount[key_triplet][key_relation2] += 1
                            else:
                                dicCount[key_triplet][key_relation2] = 1
                    else:
                        if key_relation2 == "":
                            dicCount[key_triplet] = {key_relation: 1}
                        else:
                            dicCount[key_triplet] = {key_relation: 1, key_relation2: 1}

    listWrite = ["head\ttail\tcontext\trelation_type"]
    for key_triplet in dicCount.keys():
        listSortedFrequency = sorted(list(set(dicCount[key_triplet].values())))
        setValidFrequency = set()
        for freq in listSortedFrequency:
            if (freq >= threshold) & (freq > listSortedFrequency[-1] - error_range):
                setValidFrequency.add(freq)

        setValidRelation = set()
        for relation in dicCount[key_triplet].keys():
            if dicCount[key_triplet][relation] in setValidFrequency:
                setValidRelation.add(relation)
        if len(setValidRelation) == 1:
            listWrite.append(relation)
        elif len(setValidRelation) > 1:
            boolBinding = False
            boolRegulation = False
            setRelationType = set()
            setHead = set()
            for relation in setValidRelation:
                listRelation = relation.split("\t")
                setHead.add(listRelation[0])
                setRelationType.add(listRelation[3])

            if ("Binding" in setRelationType) | len(setHead) > 1:
                boolBinding = True
            elif ("Increase" in setRelationType) & ("Decrease" in setRelationType):
                boolRegulation = True

            if boolBinding:
                listRelation[-1] = "Binding"
                listWrite.append("\t".join(listRelation))
                listRelation[0], listRelation[1] = listRelation[1], listRelation[0]
            elif boolRegulation:
                listRelation[-1] = "Regulation"
                listWrite.append("\t".join(listRelation))
            else:
                listRelation[-1] = "Increase" if "Increase" in setRelationType else "Decrease"
                listWrite.append("\t".join(listRelation))

            # print(setValidRelation)
            # print(key_triplet, boolBinding, boolRegulation)
            # print(listRelation)

    writeOutput(listWrite, "../result/integrated/voted_relation.tsv")


def context_count():
    dicCount = {}
    with open("../result/integrated/voted_relation.tsv", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[2] in dicCount:
                dicCount[listInstance[2]] += 1
            else:
                dicCount[listInstance[2]] = 1

    sorted_x = sorted(dicCount.items(), reverse=True, key=lambda kv: kv[1])
    writeOutput([tup[0] + "\t" + str(tup[1]) for tup in sorted_x], "../result/integrated/context_count.tsv")


if __name__ == '__main__':
    conflict_voting()
    context_count()
    measure()
