import time
import glob
# import xml.etree.ElementTree as ET

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
    string = string.replace("\n", "");
    string = string.replace("\"", "");
    string = string.replace("\r", "");
    string = string.strip();
    # string = string.lower();
    return string;


def statistics():
    dicCount = {}
    for strResult in glob.glob("../result/raw_result/unlabeled_*.tsv"):
        with open(strResult, "r", encoding="utf-8") as fileInput:
            for strInstance in fileInput:
                strInstance = stringCleansing(strInstance)
                listInstance = strInstance.split("\t")
                if len(listInstance) > 20:
                    if listInstance[7] == listInstance[13]:
                        continue
                    if "Forward" in listInstance[21]:
                        key = "\t".join(["_".join(sorted([listInstance[7], listInstance[13]])), listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                        if key in dicCount:
                            dicCount[key] += 1
                        else:
                            dicCount[key] = 1
                    elif "Backward" in listInstance[21]:
                        key = "\t".join(["_".join(sorted([listInstance[7], listInstance[13]])), listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])
                        if key in dicCount:
                            dicCount[key] += 1
                        else:
                            dicCount[key] = 1
                    elif "NA" in listInstance[21]:
                        key1 = "\t".join(["_".join(sorted([listInstance[7], listInstance[13]])), listInstance[7], listInstance[13], listInstance[19], listInstance[21].split("|")[0]])
                        key2 = "\t".join(["_".join(sorted([listInstance[7], listInstance[13]])), listInstance[13], listInstance[7], listInstance[19], listInstance[21].split("|")[0]])
                        if key1 in dicCount:
                            dicCount[key1] += 1
                        else:
                            dicCount[key1] = 1

                        if key2 in dicCount:
                            dicCount[key2] += 1
                        else:
                            dicCount[key2] = 1
    listValidKey = []
    for key in dicCount.keys():
        # if dicCount[key] > 5:
        listValidKey.append(key)

    listWrite = ["pair\thead\ttail\tcontext\trelation_type\tfrequency"]
    for key in listValidKey:
        listWrite.append(key + "\t" + str(dicCount[key]))
    writeOutput(listWrite, "../result/integrated/count.tsv")


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


def source():
    setGeneType = set(
        ["T087", "T088", "T028", "T085", "T086",  # Genes & Molecular Sequences
         "T116", "T195", "T123", "T122", "T103", "T120", "T104", "T200", "T196", "T126",  # Chemicals & Drugs
         "T131", "T125", "T129", "T130", "T197", "T114", "T109", "T121", "T192", "T127",  # Chemicals & Drugs
         "T026"])  # ANAT|Anatomy|T026|Cell Component
         # "T115", "T110", "T118", "T119", "T124", "T111"])
    # setDiseaseType = set(["T020", "T190", "T049", "T019", "T047", "T050", "T037", "T048", "T191", "T046", "T184"])

    setGeneType = callType(setGeneType)
    # setDiseaseType = callType(setDiseaseType)

    writeMappingTable(setGeneType, "../../UMLS/geneTable_wide.tsv")
    # writeMappingTable(setDiseaseType, "../../UMLS/diseaseTable.tsv")


def callType(setSemanticType):
    setConcept = set()
    strSemanticType = '../../UMLS/MRSTY.RRF'
    with open(strSemanticType, 'r') as fileSemanticType:
        for strInstance in fileSemanticType:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("|")
            if listInstance[1] in setSemanticType:
                setConcept.add(listInstance[0])
    return setConcept


def writeMappingTable(setConcept, strOutPath):
    listSource = ["HGNC", "OMIM"]
    setSource = set(listSource)
    dicConcept = {}
    strSemanticType = '../../UMLS/MRCONSO.RRF'
    with open(strSemanticType, 'r') as fileSemanticType:
        for strInstance in fileSemanticType:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("|")
            if (listInstance[0] in setConcept) & (listInstance[11] in setSource):
                if listInstance[0] not in dicConcept:
                    dicConcept[listInstance[0]] = ["_"] * len(listSource)
                dicConcept[listInstance[0]][listSource.index(listInstance[11])] = listInstance[13]

    listWrite = ["CUI\t" + "\t".join(listSource)]
    for concept in dicConcept.keys():
        strIDseq = "\t".join(dicConcept[concept])
        if strIDseq == "_\t_":
            continue
        listWrite.append(concept + "\t" + strIDseq)

    writeOutput(listWrite, strOutPath)


def geneMapping_UMLS2ENTREZ():
    strHGNC2ENTREZ = '../mapping/HGNC2ENTREZ.tsv'
    dicHGNC2ENTREZ = {}
    dicOMIM2ENTREZ = {}
    with open(strHGNC2ENTREZ, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")

            if len(listInstance) > 2:
                if listInstance[2] != '':
                    if listInstance[0] in dicHGNC2ENTREZ:
                        dicHGNC2ENTREZ[listInstance[0]].add(listInstance[2])
                    else:
                        dicHGNC2ENTREZ[listInstance[0]] = set([listInstance[2]])

                    if len(listInstance) > 3:
                        if listInstance[3] != '':
                            if listInstance[3] in dicOMIM2ENTREZ:
                                dicOMIM2ENTREZ[listInstance[3]].add(listInstance[2])
                            else:
                                dicOMIM2ENTREZ[listInstance[3]] = set([listInstance[2]])

    listWrite = []
    strUMLS = '../mapping/UMLS2HGNC.tsv'
    with open(strUMLS, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[1] in dicHGNC2ENTREZ:
                listWrite.append(listInstance[0] + "\t" + "|".join(dicHGNC2ENTREZ[listInstance[1]]) + "\t" + "HGNC")
            elif listInstance[2].split(".")[0] in dicOMIM2ENTREZ:
                listWrite.append(listInstance[0] + "\t" + "|".join(dicOMIM2ENTREZ[listInstance[2].split(".")[0]]) + "\t" + "OMIM")

    writeOutput(listWrite, '../mapping/UMLS2ENTREZ.tsv')


def geneMapping_UMLS2BISL():
    strENTREZ2BISL = '../mapping/ENTREZ2BISL.tsv'
    dicENTREZ2BISL = {}

    with open(strENTREZ2BISL, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[1] in dicENTREZ2BISL:
                dicENTREZ2BISL[listInstance[1]].add(listInstance[0])
            else:
                dicENTREZ2BISL[listInstance[1]] = set([listInstance[0]])

    listWrite = []
    strUMLS = '../mapping/UMLS2ENTREZ.tsv'
    with open(strUMLS, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[1] in dicENTREZ2BISL:
                listWrite.append(listInstance[0] + "\t" + "|".join(dicENTREZ2BISL[listInstance[1]]))

    writeOutput(listWrite, '../mapping/UMLS2BISL.tsv')


def callDic(strDic):
    dicConcept = {}
    with open(strDic, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            dicConcept[listInstance[0]] = listInstance[1]
    return dicConcept


def BISLmapping():
    intCnt = 0
    listWrite = []
    geneDic = callDic('../mapping/geneTable.tsv')
    diseaseDic = callDic('../mapping/diseaseTable.tsv')
    with open('../result/integrated/voted_relation.tsv', "r", encoding="utf-8") as fileInput:
        fileInput.readline()  # header
        for strInstance in fileInput:
            intCnt = intCnt + 1
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")
            if (listInstance[0] in geneDic) & (listInstance[1] in geneDic) & (listInstance[2] in diseaseDic):
                listWrite.append("\t".join([geneDic[listInstance[0]], geneDic[listInstance[1]], diseaseDic[listInstance[2]], listInstance[3]]))
    print(intCnt, "->", len(listWrite))
    writeOutput(["head\ttail\tcontext\trelation_type"] + listWrite, '../result/integrated/voted_relation_BISL.tsv')


def memory_error_file_split():
    for strResult in sorted(glob.glob("../result/raw_result/unlabeled_*.tsv")):
        with open(strResult, "r", encoding="utf-8") as fileResult:
            if "CUDA out of memory" in fileResult.readlines()[0]:
                strResource = strResult.replace("result/raw_result", "resource")
                # print(strResource)
                with open(strResource, "r", encoding="utf-8") as fileResource:
                    listContents = [stringCleansing(line) for line in fileResource.readlines()]
                    writeOutput(listContents[:int(len(listContents)/2)], strResource.replace(".tsv", "_1of2.tsv"))
                    writeOutput(listContents[int(len(listContents) / 2):], strResource.replace(".tsv", "_2of2.tsv"))
                    # print(len(listContents), len(listContents[:int(len(listContents)/2)]) + len(listContents[int(len(listContents) / 2):]))


def result_sampling():
    listWrite = []
    for strResult in glob.glob("../result/raw_result/unlabeled_*.tsv"):
        strFileName = strResult.split("/")[-1].replace(".tsv", "").replace("unlabeled_", "").replace("of2", "")
        with open(strResult, "r", encoding="utf-8") as fileInput:
            strInstance = stringCleansing(fileInput.readline())
            listInstance = strInstance.split("\t")
            if len(listInstance) > 20:
                listInstance[0] = strFileName
                listWrite.append("\t".join(listInstance))
    writeOutput(sorted(listWrite), "../result/integrated/sampled_result.tsv")


if __name__ == '__main__':
    # statistics()
    # source()
    # geneMapping_UMLS2ENTREZ()
    # geneMapping_UMLS2BISL()
    # memory_error_file_split()

    conflict_voting()
    context_count()
    BISLmapping()

    # result_sampling()
    measure()
