import time
import os.path
import itertools
import glob

init_time = time.time()


def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3660)
    mins = int((dif_time - hour * 3660) / 60)
    sec = dif_time - hour * 3660 - mins * 60
    print('Processing Time:' + str(hour) + 'hour ' + str(mins) + 'min ' + str(sec) + 'sec ')


def stringClensing(string):
    string = string.replace('\n', '')
    string = string.replace('"', '')
    string = string.replace('\r', '')
    string = string.strip()
    # string = string.lower()
    return string


def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+')
    strNewRow = '\n'.join(listString)
    manipulatedData.write(strNewRow)
    manipulatedData.close()


def disease_read(strInputPath, dicSemanticType):
    dicDisease = {}
    with open(strInputPath, "r") as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")

            if listInstance[3] == "inhibit":
                continue

            # semantic type conversion (ex: 'sosy' -> 'T123')
            listInstance[6] = listInstance[6].replace("[", "").replace("]", "")
            if "," in listInstance[6]:
                listInstance[6] = "|".join([dicSemanticType[strType.strip()] for strType in listInstance[6].split(",")])
            else:
                listInstance[6] = dicSemanticType[listInstance[6]]

            key = listInstance[0] + "\t" + listInstance[1]
            # reading records with duplication check
            if key in dicDisease:
                boolNovel = True
                for i in range(len(dicDisease[key])):
                    if dicDisease[key][i][0] == listInstance[2]:
                        if dicDisease[key][i][2].lower() == listInstance[4].lower():
                            boolNovel = False
                            if dicDisease[key][i][3] > listInstance[5]:
                                dicDisease[key][i] = "\t".join(listInstance[2:])

                if boolNovel:
                    dicDisease[key].append("\t".join(listInstance[2:]))

            else:
                dicDisease[key] = ["\t".join(listInstance[2:])]

    return dicDisease


def gene_read(strInputPath):
    dicGene= {}
    with open(strInputPath, "r") as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[3] == listInstance[6]:
                continue

            # reading records with duplication check
            if listInstance[0] in dicGene:
                dicGene[listInstance[0]].append("\t".join(listInstance[1:]))

            else:
                dicGene[listInstance[0]] = ["\t".join(listInstance[1:])]

    # for key in dicGene.keys():
    #     if len(dicGene[key]) > 1:
    #         dicGeneOut[key] = dicGene[key]

    return dicGene


def disease_read2(strInputPath):
    dicDisease = {}
    with open(strInputPath, "r") as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            # reading records with duplication check
            key = listInstance[0]
            if key in dicDisease:
                dicDisease[key].append(listInstance[2] + "-" + listInstance[3] + "\t" + listInstance[4])

            else:
                dicDisease[key] = [listInstance[2] + "-" + listInstance[3] + "\t" + listInstance[4]]
    return dicDisease

def G2D1():
    strMainPath = "/data4/jaeh/context/unlabeled/"
    # dicSemanticType = {}
    # with open("/data4/jaeh/context/UMLS2vec/UMLS_metathesaurus/semantic_type.tsv", "r") as fileInput:
    #     for strInstance in fileInput:
    #         strInstance = stringClensing(strInstance)
    #         listInstance = strInstance.split("\t")
    #         if len(listInstance) > 1:
    #             dicSemanticType[listInstance[0]] = listInstance[1]

    for idx in range(1000):
        strOutPath = strMainPath + f"unlabeled_G2D1_bannered/unlabeled_{idx + 1}.tsv"
        if os.path.isfile(strOutPath):
            print("Existing:", strOutPath)
            continue

        strDiseaseInput = strMainPath + f"1.bannered_Disease/Disease_BANNER_Sentences_Splitted{idx + 1}.tsv"
        if not os.path.isfile(strDiseaseInput):
            continue

        strGeneInput = strMainPath + f"2.unlabeled_G2/G2_{idx + 1}.tsv"
        if not os.path.isfile(strGeneInput):
            continue

        listWrite = []
        intID = 0
        print("Processing: ", idx + 1)
        dicDisease = disease_read2(strDiseaseInput)
        dicGene = gene_read(strGeneInput)
        for key in dicGene.keys():
            if key in dicDisease:
                # print(intID, key, dicGene[key])
                for strGene in dicGene[key]:
                    for strDisease in dicDisease[key]:
                        # print(intID, key, "\t".join(["\t".join(tupGene) for tupGene in listGenePair]), "\t".join(listDisease[1:]))
                        listWrite.append(str(intID) + "\tunlabeled\t" + key + "\t" + strGene + "\t" + strDisease)
                        intID += 1
        if len(listWrite):
            writeOutput(listWrite, strOutPath)


def dicDiseaseCall():
    dic = {}
    strPath = "/data4/jaeh/context/unlabeled/unlabeled_G2D1"
    for strFile in glob.glob(strPath + "/*.tsv"):
        with open(strFile, "r") as fileInput:
            for strInstance in fileInput:
                strInstance = stringClensing(strInstance)
                listInstance = strInstance.split("\t")
                dic[listInstance[11]] = "\t".join(listInstance[12:])

    return dic


def normalization_test():
    dicDisease = dicDiseaseCall()
    for strFile in glob.glob('/data4/jaeh/context/unlabeled/unlabeled_G2D1_bannered/*.tsv'):
        listWrite = []
        print("Processing: ", strFile)
        with open(strFile, 'r') as fileCorpus:
            for strInstance in fileCorpus:
                strInstance = stringClensing(strInstance)
                listInstance = strInstance.split("\t")
                if listInstance[11] in dicDisease:
                    listWrite.append(strInstance + "\t" + dicDisease[listInstance[11]])
                else:
                    listWrite.append(strInstance + "\tNA\tNA\tNA")
        writeOutput(listWrite, strFile.replace("/unlabeled_G2D1_bannered/", "/unlabeled_G2D1_bannered_normalized_by_dic/"))


if __name__ == '__main__':
    # G2D1()
    normalization_test()
    measure()
