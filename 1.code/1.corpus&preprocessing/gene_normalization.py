'''
Created by wchwang at 7/20/17

Edited 2018/10/22 sylee:
  id,pw for authentication
  get_cid > UMLS id + UMLS name

'''


#################################################################################
# usage of the script
# usage: python search-terms.py -k APIKEY -v VERSION -s STRING
# see https://documentation.uts.nlm.nih.gov/rest/search/index.html for full docs
# on the /search endpoint
#################################################################################
from __future__ import print_function
from Authentication import *
import requests
import json
import argparse
import sys
import time
import glob
from os import path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


init_time = time.time();


def measure():
	global init_time
	after_time = time.time()
	dif_time = after_time - init_time
	hour = int(dif_time / 3660)
	mins = int((dif_time - hour * 3660) / 60)
	sec = dif_time - hour * 3660 - mins * 60
	print('Processing Time:' + str(hour) + 'hour ' + str(mins) + 'min ' + str(sec) + 'sec ')


def stringClensing(string):
	string = string.replace('\n', '');
	string = string.replace('\"', '');
	string = string.replace('\r', '');
	string = string.strip();
	# string = string.lower();
	return string;


def writeOutput(listString, strOutputName):
	manipulatedData = open(strOutputName, 'w+');
	strNewRow = '\n'.join(listString);
	manipulatedData.write(strNewRow);
	manipulatedData.close();


def get_cid(search_term, dicSemanticType):
    # parser = argparse.ArgumentParser(description='process user given parameters')
    # parser.add_argument("-u", "--username", required =  True, dest="username", help = "enter username")
    # parser.add_argument("-p", "--password", required =  True, dest="password", help = "enter passowrd")
    # parser.add_argument("-k", "--apikey", required=True, dest="apikey", help="enter api key from your UTS Profile")
    # parser.add_argument("-v", "--version", required=False, dest="version", default="current",
    #                     help="enter version example-2015AA")
    # parser.add_argument("-s", "--string", required=True, dest="string",
    #                     help="enter a search term, like 'diabetic foot'")

    # args = parser.parse_args()
    # username = args.username
    # password = args.password
    # apikey = args.apikey
    # version = args.version
    # string = args.string

    # option 2
    username = 'quigmire'
    password = 'Bislaprom3#'
    # apikey = args.apikey
    # version = '2018AB'
    version = 'current'
    string = search_term

    url = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/" + version
    ##get at ticket granting ticket for the session
    tries = 0
    while 1:
        try:
            AuthClient = Authentication(username, password)
            tgt = AuthClient.gettgt()
        except requests.exceptions.ConnectionError:
            print("ConnectionError1")
            time.sleep(120)
            tries += 1
            print("retring: ", tries)
            continue
        except IndexError:
            print("IndexError1")
            time.sleep(120)
            tries += 1
            print("retring: ", tries)
            continue
        except requests.exceptions.SSLError:
            print("SSLError1")
            time.sleep(120)
            tries += 1
            print("retring: ", tries)
            continue
        break
    listResult = []
    # listSearchType = ['exact', 'words', 'approximate']
    listSearchType = ['exact']
    for searchType in listSearchType:
        boolNextPage = True
        pageNumber = 0
        # iterating through each page
        while boolNextPage & (pageNumber < 2):
            ##generate a new service ticket for each page if needed
            tries = 0
            while 1:
                try:
                    ticket = AuthClient.getst(tgt)
                except requests.exceptions.ConnectionError:
                    print("ConnectionError2")
                    time.sleep(120)
                    tries += 1
                    print("retring: ", tries)
                    continue
                except IndexError:
                    print("IndexError2")
                    time.sleep(120)
                    tries += 1
                    print("retring: ", tries)
                    continue
                except requests.exceptions.SSLError:
                    print("SSLError2")
                    time.sleep(120)
                    tries += 1
                    print("retring: ", tries)
                    continue
                break
            pageNumber += 1
            query = {'string': string, 'ticket': ticket, 'pageNumber': pageNumber, 'searchType': searchType}
            # query['includeObsolete'] = 'true'
            # query['includeSuppressible'] = 'true'
            # query['returnIdType'] = "sourceConcept"
            # query['sabs'] = "SNOMEDCT_US"

            # https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
            session = requests.Session()
            retry = Retry(connect=300, backoff_factor=15)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            # r = session.get(url + content_endpoint, params=query)

            # r = requests.get(url + content_endpoint, params=query)
            # r.encoding = 'utf-8'
            # items = json.loads(r.text)

            # https://stackoverflow.com/questions/16573332/jsondecodeerror-expecting-value-line-1-column-1-char-0
            tries = 0
            while 1:
                try:
                    items = session.get(url + content_endpoint, params=query).json()
                # items = requests.get(url + content_endpoint, params=query).json()
                except json.decoder.JSONDecodeError:
                    print("JSONDecodeError3")
                    time.sleep(120)
                    tries += 1
                    print("retring: ", tries)
                    continue
                except requests.exceptions.SSLError:
                    print("SSLError3")
                    time.sleep(120)
                    tries += 1
                    print("retring: ", tries)
                    continue
                break

            jsonData = items["result"]
            for result in jsonData["results"]:
                try:
                    # print("ui: " + result["ui"])
                    if (result["ui"] != 'NONE') & (result["ui"] in dicSemanticType) & (not 'wt Allele' in result["name"]):
                        listResult.append(result["ui"] + "\t" + result["name"] + "\t" + dicSemanticType[result["ui"]])
                        # print(string + "\t" + result["ui"] + "\t" + result["name"] + "\t" + dicSemanticType[result["ui"]])
                except:
                    print("Name Error: ", string)
                    NameError


            # if either our search returned nothing, or we're at the end
            if jsonData["results"][0]["ui"] == "NONE":
                boolNextPage = False

        # if more exact search result is available
        if len(listResult) != 0:
            break

    # if the results from all search types are not available
    if len(listResult) == 0:
        listResult = ["NA\tNA\tNA"]

    listReturn = []
    for i in range(len(listResult)):
        if "T087" in listResult[i]:
            listReturn.append(listResult[i])

    if len(listReturn) == 0:
        listReturn = listResult

    listTemp = sorted(listReturn)[0].split("\t")
    if not len(listTemp):
        print(string, listTemp, searchType)

    return "\t".join([listTemp[1], listTemp[0], listTemp[2]])


def callDic(setSemanticType):
    dicSemanticType = {}
    strSemanticType = '/data4/jaeh/context/unlabeled/MRSTY.RRF'
    with open(strSemanticType, 'r') as fileSemanticType:
        for strInstance in fileSemanticType:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("|")
            if listInstance[1] in setSemanticType:
                if listInstance[0] in dicSemanticType:
                    dicSemanticType[listInstance[0]] += ("|" + listInstance[1])
                else:
                    dicSemanticType[listInstance[0]] = listInstance[1]
    return dicSemanticType


def UMLSsearch():
    strInputPath = '/data4/jaeh/context/unlabeled/unlabeledG2D1'
    strOutputPath = '/data4/jaeh/context/unlabeled/geneNormalization_dictionary_post.tsv'

    #var1 = sys.argv[1]
    #search_term = var1.split(',')

    #inputfile = open('simple_ED_phenotype_list.txt', 'r')
    setSearchTerm = set([])
    with open(strInputPath,'r') as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            setSearchTerm.add(listInstance[5].strip())
            setSearchTerm.add(listInstance[8].strip())

    listWrite = []

    for term in sorted(list(setSearchTerm)):
        print('searching UMLS ......' + term)

        listResult = get_cid(term)
        listAppend = []
        for i in range(len(listResult)):
            if "Amino Acid, Peptide, or Protein" in listResult[i]:
                listAppend.append(listResult[i])

        if len(listAppend):
            listWrite += listAppend
        else:
            listWrite += listResult

    writeOutput(sorted(list(set(listWrite))), strOutputPath)
    print('UMLS mapping Completed \n')
    print('Output file :', strOutputPath)


def normalization():
    strCorpusPath = '/data4/jaeh/context/Resources/copus/PC13GE11GE13_valid.tsv'
    strDicPath = '/data4/jaeh/context/Resources/copus/protein_normalization_post.tsv'
    # strDicManualPath = '/data4/jaeh/context/Resources/copus/gene_dictionary_manual.tsv'

    dicGene = {}
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[0] in dicGene:
                print(listInstance[0])
            else:
                dicGene[listInstance[0]] = [listInstance[2], listInstance[1], listInstance[3]]

    listWrite = []
    with open(strCorpusPath, 'r') as fileCorpus:
        for strInstance in fileCorpus:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if (listInstance[5] in dicGene):
                if (listInstance[8] in dicGene):
                    listWrite.append("\t".join(listInstance[:6] + dicGene[listInstance[5]] + listInstance[6:9] + dicGene[listInstance[8]] + listInstance[9:]))
                else:
                    print("KeyError: " + listInstance[8])
            else:
                print("KeyError: " + listInstance[5])
    writeOutput(listWrite, strCorpusPath.replace(".tsv", "_normalized.tsv"))


def stat():
    strCorpusPath = '/data4/jaeh/context/Resources/copus/PC13GE11GE13_valid.tsv'
    dicCount = {}
    strBuffer5 = ''
    strBuffer8 = ''
    with open(strCorpusPath, 'r') as fileCorpus:
        for strInstance in fileCorpus:

            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if int(listInstance[0]) > 0:

                if strBuffer5 != listInstance[5]:
                    if listInstance[5] in dicCount:
                        dicCount[listInstance[5]] += 1
                    else:
                        dicCount[listInstance[5]] = 1

                if strBuffer8 != listInstance[8]:
                    if listInstance[8] in dicCount:
                        dicCount[listInstance[8]] += 1
                    else:
                        dicCount[listInstance[8]] = 1

                strBuffer5 = listInstance[5]
                strBuffer8 = listInstance[8]

    listSort = []
    for strKey in dicCount:
        listSort.append((dicCount[strKey], strKey))

    listWrite = []
    for pair in sorted(listSort):
        listWrite.append(str(pair[0]) + "\t" + pair[1])

    writeOutput(listWrite, '/data4/jaeh/context/Resources/copus/gene_count_distinct.tsv')


def queryClensing(string):
    setHead = set(["(", "/", "\'", ",", "-"])
    setTail = set(["/", "\'"])
    if string[0] in setHead:
        string = string[1:]
    if len(string) > 0:
        if string[-1] in setTail:
            string = string[:-1]

    string = string.strip()
    return string


def normalization_pretained():
    # strCorpusPath = '../result/PC13GE11GE13_pretrained.tsv'
    strCorpusPath = '../result/EVEX_pretrained.tsv'
    strDicPath = '../result/protein_normalization_post.tsv'
    dicSemanticType = callDic()
    dicGene = {}
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[0] in dicGene:
                print(listInstance)
            else:
                # dicGene[listInstance[0]] = listInstance[2] + "\t" + listInstance[1] + "\t" + listInstance[3]
                dicGene[listInstance[0]] = '\t'.join(listInstance[1:])

    dicLength = len(dicGene)
    listWrite = []
    print("reading...")
    intLineCnt = 1
    intQueryCnt = 1
    with open(strCorpusPath, 'r') as fileCorpus:
        for strInstance in fileCorpus:
            if intLineCnt % 100 == 0:
                if len(dicGene) > dicLength:
                    dicLength = len(dicGene)
                    listDicWrite = [key + "\t" + dicGene[key] for key in dicGene.keys()]
                    writeOutput(sorted(listDicWrite), strDicPath)

            if intLineCnt % 1000 == 0:
                print(intLineCnt)
            intLineCnt += 1

            strInstance = stringClensing(strInstance)

            listInstance = strInstance.split("\t")
            strFirstGene = queryClensing(listInstance[5])
            strSecondGene = queryClensing(listInstance[8])

            if (not strFirstGene) | (not strSecondGene):
                continue

            try:
                if strFirstGene in dicGene:
                    strFirstGeneInfo = dicGene[strFirstGene]
                else:
                    strFirstGeneInfo = get_cid(strFirstGene, dicSemanticType)
                    dicGene[strFirstGene] = strFirstGeneInfo
                    intQueryCnt += 1

                if strSecondGene in dicGene:
                    strSecondGeneInfo = dicGene[strSecondGene]
                else:
                    strSecondGeneInfo = get_cid(strSecondGene, dicSemanticType)
                    dicGene[strSecondGene] = strSecondGeneInfo
                    intQueryCnt += 1

                listWrite.append("\t".join(listInstance[:6] + [strFirstGeneInfo] + listInstance[6:9] + [strSecondGeneInfo] + listInstance[9:]))


            except():
                print("something has gone wrong...")
                break

    print("writing...")
    writeOutput(listWrite, strCorpusPath.replace(".tsv", "_normalized.tsv"))


def normalization_unlabeled():
    strCorpusPath = '/data4/jaeh/context/unlabeled'
    strDicPath = strCorpusPath + '/protein_normalization.tsv'
    setGeneType = set(
        ["T087", "T088", "T028", "T085", "T086", "T116", "T195", "T123", "T122", "T118", "T103", "T120", "T104", "T200",
         "T111", "T196", "T126", "T131", "T125", "T129", "T130", "T197", "T119", "T124", "T114", "T109", "T115", "T121",
         "T192", "T110", "T127", "T026"])
    setDiseaseType = set(["T020", "T190", "T049", "T019", "T047", "T050", "T037", "T048", "T191", "T046", "T184" ])


    dicDisease = callDic(setDiseaseType)
    dicGene = callDic(setGeneType)

    dicGene = {}
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[0] in dicGene:
                print(listInstance)
            else:
                # dicGene[listInstance[0]] = listInstance[2] + "\t" + listInstance[1] + "\t" + listInstance[3]
                dicGene[listInstance[0]] = '\t'.join(listInstance[1:])

    dicLength = len(dicGene)

    for strFile in glob.glob(strCorpusPath + "/unlabeled_G2D1/*.tsv"):
        strOut = strFile.replace("unlabeled_G2D1", "unlabeled_G2D1_normalized")
        if path.isfile(strOut):
            print("Existing: ", strFile)
            continue

        listWrite = []
        print("Processing: ", strFile)
        intLineCnt = 1
        intQueryCnt = 1
        with open(strFile, 'r') as fileCorpus:
            for strInstance in fileCorpus:
                if intLineCnt % 10 == 0:
                    if len(dicGene) > dicLength:
                        dicLength = len(dicGene)
                        listDicWrite = [key + "\t" + dicGene[key] for key in dicGene.keys()]
                        writeOutput(sorted(listDicWrite), strDicPath)

                if intLineCnt % 1000 == 0:
                    print(intLineCnt)
                intLineCnt += 1

                strInstance = stringClensing(strInstance)

                listInstance = strInstance.split("\t")
                strFirstGene = queryClensing(listInstance[5])
                strSecondGene = queryClensing(listInstance[8])

                if (not strFirstGene) | (not strSecondGene):
                    continue

                try:
                    if strFirstGene in dicGene:
                        strFirstGeneInfo = dicGene[strFirstGene]
                    else:
                        strFirstGeneInfo = get_cid(strFirstGene, dicDisease)
                        dicGene[strFirstGene] = strFirstGeneInfo
                        intQueryCnt += 1

                    if strSecondGene in dicGene:
                        strSecondGeneInfo = dicGene[strSecondGene]
                    else:
                        strSecondGeneInfo = get_cid(strSecondGene, dicDisease)
                        dicGene[strSecondGene] = strSecondGeneInfo
                        intQueryCnt += 1

                    listWrite.append("\t".join(listInstance[:6] + [strFirstGeneInfo] + listInstance[6:9] + [strSecondGeneInfo] + listInstance[9:]))

                except():
                    print("something has gone wrong...")
                    break

        writeOutput(listWrite, strOut)


def id_conversion():
    dicName2ID = {}
    dicAbbr2ID = {}
    # strInputPath = "../result/PC13GE11GE13_normalized.tsv"
    # strInputPath = "../result/PC13GE11GE13_pretrained_normalized.tsv"
    strInputPath = "../result/EVEX_pretrained_normalized.tsv"

    # line = 0
    with open("../../UMLS2vec/resource/semantic/SRDEF", "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            # line += 1
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("|")
            dicName2ID[listInstance[2]] = listInstance[1]
            dicAbbr2ID[listInstance[8]] = listInstance[1]
            # print(line, listInstance[8])

    # line = 0
    listWrite = []
    with open(strInputPath, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            # line += 1
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")

            if listInstance[8] != "NA":
                list1 = listInstance[8].split('|')
                listInstance[8] = '|'.join([dicName2ID[name] for name in list1])

            if listInstance[14] != "NA":
                list2 = listInstance[14].split('|')
                listInstance[14] = '|'.join([dicName2ID[name] for name in list2])

            # listPhen = listInstance[20].replace("[", "").replace("]", "").split(",")
            # listInstance[20] = '|'.join([dicAbbr2ID[abbr.strip()] for abbr in listPhen])

            listWrite.append("\t".join(listInstance))

    writeOutput(listWrite,
                strInputPath.replace("_normalized", "_IDconverted").replace("../result", "../../learning/resource"))


def minor_touch():
    # column arrangement
    import re
    pattern = re.compile('C\d\d\d\d\d\d\d')
    listWrite = []
    strDicPath = '../result/protein_normalization_post.tsv'
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")

            if pattern.match(listInstance[1]):
                listInstance[1], listInstance[2] = listInstance[2], listInstance[1]
                listWrite.append("\t".join(listInstance))
            elif pattern.match(listInstance[2]):
                listWrite.append(strInstance)
            elif listInstance[1] == "NA":
                listWrite.append(strInstance)
            else:
                print(strInstance)

    writeOutput(sorted(listWrite), strDicPath)


def listCompletion():
    # stopword reading in
    setStopWord = set()
    strStopWordPath = '../result/stopwords.tsv'
    with open(strStopWordPath, 'r') as fileStopWord:
        for strInstance in fileStopWord:
            # strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            setStopWord.add(listInstance[0])

    # # manual result reading in
    # setManual = set()
    # strStopWordPath = '../result/PC13GE11GE13_pretrained.tsv'
    # with open(strStopWordPath, 'r') as fileStopWord:
    #     for strInstance in fileStopWord:
    #         strInstance = stringClensing(strInstance)
    #         listInstance = strInstance.split("\t")
    #         setManual.add(listInstance[5])
    #         setManual.add(listInstance[11])

    listWrite = []
    dicCnt = {'exact': 0, 'words': 0, 'approximate': 0}
    strDicPath = '../result/protein_normalization_clear.tsv'
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if not listInstance[0] in setStopWord:
                if listInstance[1] == "NA":
                    result, searchType = get_cid(listInstance[0])
                    strInstance = listInstance[0] + "\t" + result
                    dicCnt[searchType] += 1
                    print(strInstance, searchType, dicCnt[searchType])
                else:
                    print(listInstance[0])

            # if not listInstance[0] in setManual:
            #     strInstance = listInstance[0] + "\tNA\tNA\tNA"
            listWrite.append(strInstance)
    writeOutput(listWrite, strDicPath.replace("_clear.tsv", "_post.tsv"))


def callDictionary(strDicPath):
    dic = {}
    with open(strDicPath, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            if listInstance[0] not in dic:
                dic[listInstance[0]] = '\t'.join(listInstance[1:])
    return dic


def bannered_normalization_unlabeled():
    strCorpusPath = '/data4/jaeh/context/unlabeled/unlabeled_G2D1_bannered_normalized_by_dic'
    strGeneDicPath = '/data4/jaeh/context/unlabeled/protein_normalization.tsv'
    strDiseaseDicPath = '/data4/jaeh/context/unlabeled/disease_normalization.tsv'
    setGeneType = set(
        ["T087", "T088", "T028", "T085", "T086", "T116", "T195", "T123", "T122", "T118", "T103", "T120", "T104", "T200",
         "T111", "T196", "T126", "T131", "T125", "T129", "T130", "T197", "T119", "T124", "T114", "T109", "T115", "T121",
         "T192", "T110", "T127", "T026"])
    setDiseaseType = set(["T020", "T190", "T049", "T019", "T047", "T050", "T037", "T048", "T191", "T046", "T184"])

    dicDiseaseType = callDic(setDiseaseType)
    # print(len(dicDiseaseType))
    dicGeneType = callDic(setGeneType)
    # print(len(dicGeneType))

    dicDisease = callDictionary(strDiseaseDicPath)
    dicGene = callDictionary(strGeneDicPath)

    dicLength = len(dicGene)
    for strFile in glob.glob(strCorpusPath + "/*.tsv"):
        temp_time = time.time()
        strOut = strFile.replace("unlabeled_G2D1_bannered_normalized_by_dic", "unlabeled_G2D1_bannered_normalized")
        if path.isfile(strOut):
            # print("Existing: ", strOut)
            continue

        listWrite = []
        print("Processing: ", strFile)
        intLineCnt = 1
        intQueryCnt = 1
        with open(strFile, 'r') as fileCorpus:
            for strInstance in fileCorpus:
                if intLineCnt % 10 == 0:
                    if len(dicGene) > dicLength:
                        dicLength = len(dicGene)
                        listGeneDicWrite = [key + "\t" + dicGene[key] for key in dicGene.keys()]
                        listDiseaseDicWrite = [key + "\t" + dicDisease[key] for key in dicDisease.keys()]
                        writeOutput(sorted(listGeneDicWrite), strGeneDicPath)
                        writeOutput(sorted(listDiseaseDicWrite), strDiseaseDicPath)

                if intLineCnt % 10000 == 0:
                    print(intLineCnt)
                intLineCnt += 1

                strInstance = stringClensing(strInstance)

                listInstance = strInstance.split("\t")
                strFirstGene = queryClensing(listInstance[5])
                strSecondGene = queryClensing(listInstance[8])

                if (strFirstGene == "") | (strSecondGene == ""):
                    continue

                if strFirstGene in dicGene:
                    strFirstGeneInfo = dicGene[strFirstGene]
                else:
                    strFirstGeneInfo = get_cid(strFirstGene, dicGeneType)
                    dicGene[strFirstGene] = strFirstGeneInfo
                    intQueryCnt += 1

                if strSecondGene in dicGene:
                    strSecondGeneInfo = dicGene[strSecondGene]
                else:
                    strSecondGeneInfo = get_cid(strSecondGene, dicGeneType)
                    dicGene[strSecondGene] = strSecondGeneInfo
                    intQueryCnt += 1

                strDiseaseInfo = "\t".join(listInstance[12:])
                if listInstance[13] == "NA":
                    strDisease = queryClensing(listInstance[11])
                    if strDisease == "":
                        continue
                    if strDisease in dicDisease:
                        strDiseaseInfo = dicDisease[strDisease]
                    else:
                        strDiseaseInfo = get_cid(strDisease, dicDiseaseType)
                        dicDisease[strDisease] = strDiseaseInfo
                listWrite.append("\t".join(listInstance[:6] + [strFirstGeneInfo] + listInstance[6:9] + [strSecondGeneInfo] + listInstance[9:12] + [strDiseaseInfo]))
        writeOutput(listWrite, strOut)
        temp_measure(temp_time)


def temp_measure(temp_time):
    after_time = time.time()
    dif_time = after_time - temp_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


def dicIDconversion():
    dicSemanticType = {}
    strSemanticType = '/data4/jaeh/context/unlabeled/MRSTY.RRF'
    with open(strSemanticType, 'r') as fileSemanticType:
        for strInstance in fileSemanticType:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("|")
            dicSemanticType[listInstance[3]] = listInstance[1]
    print(len(dicSemanticType))

    strFile = "/data4/jaeh/context/unlabeled/protein_normalization_backup.tsv"
    listWrite = []
    with open(strFile, 'r') as fileDic:
        for strInstance in fileDic:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            listInstance[3] = "|".join([dicSemanticType[name] for name in listInstance[3].split("|")]) if "NA" != listInstance[3] else listInstance[3]
            listWrite.append("\t".join(listInstance))
    writeOutput(listWrite, strFile.replace("_backup", ""))


if __name__ == '__main__':
    # normalization_pretained()
    # id_conversion()
    # minor_touch()
    # listCompletion()
    # normalization_unlabeled()
    # get_cid("BRCA1", 1)
    bannered_normalization_unlabeled()
    # dicIDconversion()
    # measure()
