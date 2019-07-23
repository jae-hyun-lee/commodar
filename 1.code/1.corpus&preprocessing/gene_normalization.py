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
	string = string.replace('\n', '')
	string = string.replace('\"', '')
	string = string.replace('\r', '')
	string = string.strip()
	return string


def writeOutput(listString, strOutputName):
	manipulatedData = open(strOutputName, 'w+')
	strNewRow = '\n'.join(listString)
	manipulatedData.write(strNewRow)
	manipulatedData.close()


def get_cid(search_term, dicSemanticType):
    username = 'YOUR ID'
    password = 'YOUR PASSWORD'
    version = '2018AB'
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
            query = {'string': search_term, 'ticket': ticket, 'pageNumber': pageNumber, 'searchType': searchType}

            # https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
            session = requests.Session()
            retry = Retry(connect=300, backoff_factor=15)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

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
                    print("Name Error: ", search_term)
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
        print(search_term, listTemp, searchType)

    return "\t".join([listTemp[1], listTemp[0], listTemp[2]])


def callDic(setSemanticType):
    dicSemanticType = {}
    strSemanticType = 'MRSTY.RRF'
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


def temp_measure(temp_time):
    after_time = time.time()
    dif_time = after_time - temp_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


if __name__ == '__main__':
    bannered_normalization_unlabeled()
    measure()
