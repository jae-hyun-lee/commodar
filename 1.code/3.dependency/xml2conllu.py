# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import sys
import time
import collections
import gzip
import glob
from os import path

TokenInfo=collections.namedtuple("TokenInfo","id,form,POS,heads,deprels")
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


def get_id(s):
    """From something like bt_0 return 0"""
    return int(s.rsplit("_",1)[1])


def get_deps(ti):
    """Given token info, pick one head, one deprel, and stick rest to deps. Return head, deprel, deps"""
    all_deps=list(zip(ti.heads,ti.deprels))
    all_deps.sort(key=lambda h_deprel:(abs(ti.id-h_deprel[0]),h_deprel[1])) #sort by closest first, then alphabetically
    if not all_deps: #attach to root
        return -1,"root","_"
    head,deprel=all_deps[0]
    all_deps=all_deps[1:]
    all_deps.sort() #deps must be sorted by head by CoNLL-U spec
    if not all_deps:
        deps="_"
    else:
        deps="|".join("{0}:{1}".format(h+1,drel) for h,drel in all_deps)
    return head,deprel,deps


def doc2conllu(doc_elem):
    listReturn = []
    for sid,sent_elem in enumerate(doc_elem):
        # list of TokenInfo tuples
        tokens=[TokenInfo(id,token_elem.get("text"),token_elem.get("POS"),[],[]) for id,token_elem in enumerate(sent_elem.findall("analyses/tokenization/token"))]
        for dep_elem in sent_elem.findall("analyses/parse/dependency"):
               gov,dep,deprel=get_id(dep_elem.get("t1")),get_id(dep_elem.get("t2")),dep_elem.get("type")
               tokens[dep].heads.append(gov)
               tokens[dep].deprels.append(deprel)
        if tokens:
            #So now I should have all I need
            listReturn.append("#pmid.sid {0}.{1}".format(doc_elem.get("origId"),sid))
            for tok in tokens:
                head,deprel,deps=get_deps(tok)
                listReturn.append('\t'.join([str(tok.id+1), tok.form, "_", tok.POS, "_", "_", str(head+1), deprel, deps, "_"]))
            listReturn.append('')
    return listReturn


def extractionGz():
    strMainPath = "/data4/jaeh/context/word2vecf"
    listGZ_2014 = glob.glob(strMainPath + "/TEES_processed_2014/44gb/*.xml.gz")
    listInputHead = list(set([strGZ[:69] for strGZ in listGZ_2014]))

    for strInput in sorted(listInputHead):
        outfile = strInput.replace("TEES_processed_2014/44gb/", "resource/pubmed_parse/") + ".conll"
        if path.isfile(outfile):
            print("Existing: ", outfile)
            continue

        print("processing: ", outfile)
        listWrite = []
        listInput = glob.glob(strInput + "*.xml.gz")
        for strGZ in listInput:
            fileGZ = gzip.open(strGZ, 'rb')
            print(strGZ)
            # for (event,elem) in ET.iterparse(sys.stdin,["end"]):
            for (event, elem) in ET.iterparse(fileGZ):
                if elem.tag != "document":
                    continue
                # We have a document tag done
                listWrite += doc2conllu(elem)

        writeOutput(listWrite, outfile)


def xml2txt_extractionGz():
    listWrite = []
    strMainPath = "/data4/jaeh/context/word2vecf"
    #listGZ_2015 = glob.glob(strMainPath + "/TEES_processed/*.xml.gz")
    #for strGZ in listGZ_2015:
    #    print(strGZ)
    #    fileGZ = gzip.open(strGZ, 'rb')
    #    # for (event,elem) in ET.iterparse(sys.stdin,["end"]):
    #    for (event, elem) in ET.iterparse(fileGZ):
    #        if elem.tag == "sentence":
    #            listWrite.append(elem.get("text"))

    listGZ_2014 = glob.glob(strMainPath + "/TEES_processed_2014/*.xml.gz")
    for strGZ in listGZ_2014:
        print(strGZ)
        fileGZ = gzip.open(strGZ, 'rb')
        # for (event,elem) in ET.iterparse(sys.stdin,["end"]):
        for (event, elem) in ET.iterparse(fileGZ):
            if elem.tag == "sentence":
                listWrite.append(elem.get("text"))

    #listXML = glob.glob(strMainPath + "/GE,PC/*.xml")
    #for strXML in listXML:
    #    print(strXML)
    #    for (event, elem) in ET.iterparse(strXML):
    #        if elem.tag == "sentence":
    #            listWrite.append(elem.get("text"))

    writeOutput(listWrite, strMainPath + "/resource/corpus(+GE,PC).txt")


if __name__=="__main__":
    extractionGz()
    # xml2txt_extractionGz()
    measure()
