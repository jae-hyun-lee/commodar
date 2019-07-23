import time
import subprocess
import os
import shutil
import glob
import os.path
from multiprocessing import Process

init_time = time.time()
intCore = 3

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
    return string


def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+')
    strNewRow = '\n'.join(listString)
    manipulatedData.write(strNewRow)
    manipulatedData.close()


def removeTempFiles(strOutputFileName):
    os.remove(strOutputFileName)


def abstractSplit(listSentence, strProcessingFile):

    nTotalSentence = len(listSentence)
    nSentenceIndenx = 0
    nSubFileIndex = 0
    listSentencePerFile = []
    listInputFileName_ForBANNER = []

    for instance in listSentence:
        nSentenceIndenx += 1
        if nSentenceIndenx <= nTotalSentence / intCore:
            listSentencePerFile.append(instance)
        else:
            nSentenceIndenx = 0
            nSubFileIndex += 1

            strSubFileName = 'tempinput/Temp_' + strProcessingFile + '_' + str(nSubFileIndex) + '.tsv'
            listInputFileName_ForBANNER.append(strSubFileName)
            writeOutput(listSentencePerFile, strSubFileName)

            listSentencePerFile = []
            listSentencePerFile.append(instance)

    ### Remained sentences ###
    nSubFileIndex += 1
    strSubFileName = 'tempinput/Temp_' + strProcessingFile + '_' + str(nSubFileIndex) + '.tsv'
    listInputFileName_ForBANNER.append(strSubFileName)
    writeOutput(listSentencePerFile, strSubFileName)

    return listInputFileName_ForBANNER


def manipulationForInput(fileName, strProcessingFile):
    listSentence = []
    for instance in fileName:
        strLineFile = stringClensing(instance)
        listLineToken = strLineFile.split('\t')
        if len(listLineToken) > 1:
            strSentenceID = listLineToken[0];
            strSentence = listLineToken[1];

            listNewRow = [strSentenceID, strSentence];
            strNewRow = '\t'.join(listNewRow);

            listSentence.append(strNewRow);

    fileName.close()

    listInputFileName_ForBANNER = abstractSplit(listSentence, strProcessingFile)

    return listInputFileName_ForBANNER


def entityTaggingUsingBANNER(strTaggingType, strInputFileName_ForBANNER, strProcessingFile, strTempDirectoryName):
    strInputFileName = strInputFileName_ForBANNER
    strSubFileIndex = strInputFileName.rsplit('_', 1)[1].replace('.tsv', '')
    strOutputFileName = strTempDirectoryName + '/' + strTaggingType + '_' + strProcessingFile + '_' + strSubFileIndex + '.txt'
    if strTaggingType == 'Gene':
        strConfigFileName = 'config/banner_BC2GM.xml'
    else:
        strConfigFileName = 'config/banner_NCBIDisease_TRAIN.xml'

    strCommand_BANNER = './banner.sh tag ' + strConfigFileName + ' ' + strInputFileName + ' ' + strOutputFileName
    # print(strCommand_BANNER)
    subprocess.getoutput(strCommand_BANNER)
    

def tempResultsAggregation(listTempResultsFile, strOutputFile):
    listResults = []
    for tempFile in listTempResultsFile:
        fileName = open(tempFile, 'r')
        listResults += [stringClensing(instance) for instance in fileName.readlines()]
        fileName.close()

    ### Write string in file ###
    if len(listResults):
        writeOutput(listResults, strOutputFile)


def tagBANNERExcution(strProcessingFile, strInputFile, strOutputFile, strTaggingType):
    ### Manipulation input file for BANNER ###
    f_Sentences = open(strInputFile, 'r')
    listInputFileName_ForBANNER = manipulationForInput(f_Sentences, strProcessingFile)

    ### Multi-processing ###
    strTempDirectoryName = 'tempoutput/Temp_' + strProcessingFile
    if os.path.isdir(strTempDirectoryName): shutil.rmtree(strTempDirectoryName)
    os.mkdir(strTempDirectoryName)
    listProcess_BANNER = []

    for strInputFileName_ForBANNER in listInputFileName_ForBANNER:
        proc = Process(target=entityTaggingUsingBANNER,
                       args=(strTaggingType, strInputFileName_ForBANNER, strProcessingFile, strTempDirectoryName))
        proc.start()
        listProcess_BANNER.append(proc)

    ### Waiting all tools ###
    for proc in listProcess_BANNER: proc.join()

    ### Aggregation of temporary results ###
    listTempResultsFile = glob.glob(strTempDirectoryName + '/*.txt')
    tempResultsAggregation(listTempResultsFile, strOutputFile)


    ### Remove temp files ###
    for removeFileName in listInputFileName_ForBANNER: removeTempFiles(removeFileName)
    shutil.rmtree(strTempDirectoryName)


def batch_processing():
    strMainPath = "[SENTENCE TSV PATH HERE]"
    listFile = glob.glob(strMainPath + "*.tsv")
    strTaggingType = 'Gene'
    # strTaggingType = 'Disease'
    for strInputFile in listFile:
        strProcessingFile = strInputFile.replace(strMainPath, "").replace(".tsv", "")
        strOutputFile = 'bannered_' + strTaggingType + '/' + strTaggingType + '_BANNER_' + strProcessingFile + '.tsv'
        if os.path.isfile(strOutputFile):
            print("Existing file: " + strProcessingFile)
        else:
            print("Processing file: " + strProcessingFile)
            tagBANNERExcution(strProcessingFile, strInputFile, strOutputFile, strTaggingType)


### Main function ###
if __name__ == '__main__':
    batch_processing()
    measure()



