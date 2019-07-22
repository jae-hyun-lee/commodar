import scipy
import scipy.io
import random
import numpy as np

from batching import *
import time

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


def stringClensing(string):
    string = string.replace("\n", "");
    string = string.replace("\"", "");
    string = string.replace("\r", "");
    string = string.strip();
    # string = string.lower();
    return string;


def read_from_id(filename):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity


def init_norm_Vector(relinit, entinit, embedding_size):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstent.append(tmp)
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


def getID(folder):
    lstEnts = {}
    lstRels = {}
    with open(folder + 'train.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'valid.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'test.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    wri = open(folder + 'entity2id.txt', 'w')
    for entity in lstEnts:
        wri.write(entity + '\t' + str(lstEnts[entity]))
        wri.write('\n')
    wri.close()

    wri = open(folder + 'relation2id.txt', 'w')
    for entity in lstRels:
        wri.write(entity + '\t' + str(lstRels[entity]))
        wri.write('\n')
    wri.close()


def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]

    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val


def load_triples_from_txt(filename, words_indexes=None, parse_line=parse_line):
    """
    Take a list of file names and build the corresponding dictionary of triples
    """
    if words_indexes == None:
        words_indexes = dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(words_indexes)
        next_ent = max(words_indexes.values()) + 1

    data = dict()

    with open(filename) as f:
        lines = f.readlines()

    for _, line in enumerate(lines):
        sub, obj, rel, val = parse_line(line)

        if sub in entities:
            sub_ind = words_indexes[sub]
        else:
            sub_ind = next_ent
            next_ent += 1
            words_indexes[sub] = sub_ind
            entities.add(sub)

        if rel in entities:
            rel_ind = words_indexes[rel]
        else:
            rel_ind = next_ent
            next_ent += 1
            words_indexes[rel] = rel_ind
            entities.add(rel)

        if obj in entities:
            obj_ind = words_indexes[obj]
        else:
            obj_ind = next_ent
            next_ent += 1
            words_indexes[obj] = obj_ind
            entities.add(obj)

        data[(sub_ind, rel_ind, obj_ind)] = val

    indexes_words = {}
    for tmpkey in words_indexes:
        indexes_words[words_indexes[tmpkey]] = tmpkey

    return data, words_indexes, indexes_words


def build_data(name, path):
    folder = path + '/' + name + '/'

    train_triples, words_indexes, _ = load_triples_from_txt(folder + 'train.txt', parse_line=parse_line)

    valid_triples, words_indexes, _ = load_triples_from_txt(folder + 'valid.txt',
                                                            words_indexes=words_indexes, parse_line=parse_line)

    test_triples, words_indexes, indexes_words = load_triples_from_txt(folder + 'test.txt',
                                                                       words_indexes=words_indexes,
                                                                       parse_line=parse_line)

    entity2id, id2entity = read_from_id(folder + '/entity2id.txt')
    relation2id, id2relation = read_from_id(folder + '/relation2id.txt')
    left_entity = {}
    right_entity = {}

    with open(folder + 'train.txt') as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        head, tail, rel, val = parse_line(line)
        # count the number of occurrences for each (head, rel)
        if relation2id[rel] not in left_entity:
            left_entity[relation2id[rel]] = {}
        if entity2id[head] not in left_entity[relation2id[rel]]:
            left_entity[relation2id[rel]][entity2id[head]] = 0
        left_entity[relation2id[rel]][entity2id[head]] += 1
        # count the number of occurrences for each (rel, tail)
        if relation2id[rel] not in right_entity:
            right_entity[relation2id[rel]] = {}
        if entity2id[tail] not in right_entity[relation2id[rel]]:
            right_entity[relation2id[rel]][entity2id[tail]] = 0
        right_entity[relation2id[rel]][entity2id[tail]] += 1

    left_avg = {}
    # for i in range(len(relation2id)):
    for i in left_entity.keys():
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_avg = {}
    # for i in range(len(relation2id)):
    for i in left_entity.keys():
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    # for i in range(len(relation2id)):
    for i in left_entity.keys():
        headTailSelector[i] = 1000 * right_avg[i] / (right_avg[i] + left_avg[i])

    return train_triples, valid_triples, test_triples, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation


def dic_of_chars(words_indexes):
    lstChars = {}
    for word in words_indexes:
        for char in word:
            if char not in lstChars:
                lstChars[char] = len(lstChars)
    lstChars['unk'] = len(lstChars)
    return lstChars


def convert_to_seq_chars(x_batch, lstChars, indexes_words):
    lst = []
    for [tmpH, tmpR, tmpT] in x_batch:
        wH = [lstChars[tmp] for tmp in indexes_words[tmpH]]
        wR = [lstChars[tmp] for tmp in indexes_words[tmpR]]
        wT = [lstChars[tmp] for tmp in indexes_words[tmpT]]
        lst.append([wH, wR, wT])
    return lst


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    sequence_padded, sequence_length = [], []
    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return np.array(sequence_padded).astype(np.int32), np.array(sequence_length).astype(np.int32)


def split_data():
    # strInputPath = "../resource/input/semantic/"
    # strInputPath = "../resource/input/concept_partial/"
    # strInputPath = "../resource/input/concept/"
    strInputPath = "../resource/input/concept_source_or/"
    listContents = []
    with open(strInputPath + "whole.txt", "r") as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listContents.append(strInstance)

    listContents = list(set(listContents))
    intNumInstance = len(listContents)
    intTrainLen = int(0.8 * intNumInstance)
    intValidLen = int(0.1 * intNumInstance)

    from random import shuffle
    shuffle(listContents)

    listTrain = listContents[: intTrainLen]
    listValid = listContents[intTrainLen: intTrainLen + intValidLen]
    listTest = listContents[intTrainLen + intValidLen:]

    print(intNumInstance)
    print(len(listTrain))
    print(len(listValid))
    print(len(listTest))

    print(set(listTrain) & set(listValid))
    print(set(listValid) & set(listTest))
    print(set(listTest) & set(listTrain))

    writeOutput(listTrain, strInputPath + "train.txt")
    writeOutput(listValid, strInputPath + "valid.txt")
    writeOutput(listTest, strInputPath + "test.txt")


def read_data_concept():
    # # partial
    # setPartial = set()
    # strPartial = "../../learning/resource/PC13GE11GE13_valid_fullynormalized.tsv"
    # with open(strPartial, "r") as fileInput:
    #     for strInstance in fileInput:
    #         strInstance = stringClensing(strInstance)
    #         listInstance = strInstance.split("\t")
    #         setPartial = setPartial.union(set([listInstance[7], listInstance[13], listInstance[19]]))
    # print('C0275518' in setPartial)
    #
    # strConcept = '../resource/concept/MRSTY.RRF'
    # setConcept = set()
    # with open(strConcept, "r") as fileInput:
    #     for strInstance in fileInput:
    #         listInstance = stringClensing(strInstance).split("|")
    #         if listInstance[1] in setSemanticType:
    #             setConcept.add(listInstance[0])

    strRel = '../resource/concept/MRREL.RRF'
    setRel = set()
    dicRel = {}
    dicConcept = {}
    with open(strRel, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringClensing(strInstance).split("|")
            # if (listInstance[0] == 'C0275518') | (listInstance[4] == 'C0275518'):
            #     print(listInstance)

            # if (listInstance[0] in setPartial) | (listInstance[4] in setPartial):
            if not listInstance[0] in dicConcept:
                dicConcept[listInstance[0]] = len(dicConcept)

            if not listInstance[4] in dicConcept:
                dicConcept[listInstance[4]] = len(dicConcept)

            if not listInstance[3] in dicRel:
                dicRel[listInstance[3]] = len(dicRel)
            setRel.add(' '.join([listInstance[0], listInstance[3], listInstance[4]]))
    print(len(dicConcept), len(dicRel), len(setRel))

    writeOutput(sorted([strKey + ' ' + str(dicConcept[strKey]) for strKey in dicConcept.keys()]),
                "../resource/input/concept/entity2id.txt")

    writeOutput(sorted([strKey + ' ' + str(dicRel[strKey]) for strKey in dicRel.keys()]),
                "../resource/input/concept/relation2id.txt")

    writeOutput(sorted(list(setRel)), "../resource/input/concept/whole.txt")


def resource_check():
    setSource = set()
    setPartial = set()
    strPartial = "../../learning/resource/PC13GE11GE13_valid_fullynormalized.tsv"
    with open(strPartial, "r") as fileInput:
        for strInstance in fileInput:
            strInstance = stringClensing(strInstance)
            listInstance = strInstance.split("\t")
            setPartial = setPartial.union(set([listInstance[7], listInstance[13], listInstance[19]]))

    strRel = '../resource/concept/MRREL.RRF'
    # setRel = set()
    # dicRel = {}
    # dicConcept = {}
    with open(strRel, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringClensing(strInstance).split("|")
            if (listInstance[0] in setPartial) | (listInstance[4] in setPartial):
            # if (listInstance[0] in setPartial) & (listInstance[4] in setPartial):
                setSource.add(listInstance[10])
            # setSource2.add(listInstance[11])
    return setSource


def read_data_concept_sourceFIltered():
    setSource = resource_check()

    strRel = '../resource/concept/MRREL.RRF'
    setRel = set()
    dicRel = {}
    dicConcept = {}
    with open(strRel, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringClensing(strInstance).split("|")
            if listInstance[10] in setSource:
                if not listInstance[0] in dicConcept:
                    dicConcept[listInstance[0]] = len(dicConcept)

                if not listInstance[4] in dicConcept:
                    dicConcept[listInstance[4]] = len(dicConcept)

                if not listInstance[3] in dicRel:
                    dicRel[listInstance[3]] = len(dicRel)
                setRel.add(' '.join([listInstance[0], listInstance[3], listInstance[4]]))
    print(len(dicConcept), len(dicRel), len(setRel))

    writeOutput(sorted([strKey + ' ' + str(dicConcept[strKey]) for strKey in dicConcept.keys()]),
                "../resource/input/concept_source_or/entity2id.txt")

    writeOutput(sorted([strKey + ' ' + str(dicRel[strKey]) for strKey in dicRel.keys()]),
                "../resource/input/concept_source_or/relation2id.txt")

    writeOutput(sorted(list(setRel)), "../resource/input/concept_source_or/whole.txt")


def read_data_semantic():
    strType = "../resource/semantic/SRDEF"
    setType = set()
    with open(strType, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringClensing(strInstance).split("|")
            if listInstance[0] == "STY":
                setType.add(listInstance[1])

    dicType = {}
    dicRel = {}

    setRel = set()
    # strStruct = "../resource/semantic/SRSTR"
    # with open(strStruct, "r") as fileInput:
    #     for strInstance in fileInput:
    #         listInstance = stringClensing(strInstance).split("|")
    #         if (listInstance[0] in dicName2ID) & (listInstance[1] in dicName2ID) & (listInstance[2] in dicName2ID):
    #             if not dicName2ID[listInstance[0]] in dicType:
    #                 dicType[dicName2ID[listInstance[0]]] = len(dicType)
    #             if not dicName2ID[listInstance[2]] in dicType:
    #                 dicType[dicName2ID[listInstance[2]]] = len(dicType)
    #             if not dicName2ID[listInstance[1]] in dicRel:
    #                 dicRel[dicName2ID[listInstance[1]]] = len(dicRel)
    #             setRel.add(
    #                 ' '.join([dicName2ID[listInstance[0]], dicName2ID[listInstance[1]], dicName2ID[listInstance[2]]]))
    # print(len(dicType), len(dicRel), len(setRel))

    strStruct = "../resource/semantic/SRSTRE1"
    with open(strStruct, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringClensing(strInstance).split("|")
            if (listInstance[0] in setType) & (listInstance[2] in setType):
                if not listInstance[0] in dicType:
                    dicType[listInstance[0]] = len(dicType)
                if not listInstance[2] in dicType:
                    dicType[listInstance[2]] = len(dicType)
                if not listInstance[1] in dicRel:
                    dicRel[listInstance[1]] = len(dicRel)
                setRel.add(' '.join([listInstance[0], listInstance[1], listInstance[2]]))
    print(len(dicType), len(dicRel), len(setRel))

    # strStruct = "../resource/semantic/SRSTRE2"
    # with open(strStruct, "r") as fileInput:
    #     for strInstance in fileInput:
    #         listInstance = stringClensing(strInstance).split("|")
    #         if (listInstance[0] in dicName2ID) & (listInstance[1] in dicName2ID) & (listInstance[2] in dicName2ID):
    #             if not dicName2ID[listInstance[0]] in dicType:
    #                 dicType[dicName2ID[listInstance[0]]] = len(dicType)
    #
    #             if not dicName2ID[listInstance[2]] in dicType:
    #                 dicType[dicName2ID[listInstance[2]]] = len(dicType)
    #
    #             if not dicName2ID[listInstance[1]] in dicRel:
    #                 dicRel[dicName2ID[listInstance[1]]] = len(dicRel)
    #             setRel.add(
    #                 ' '.join([dicName2ID[listInstance[0]], dicName2ID[listInstance[1]], dicName2ID[listInstance[2]]]))
    # print(len(dicType), len(dicRel), len(setRel))

    writeOutput([strKey + ' ' + str(dicType[strKey]) for strKey in dicType.keys()],
                "../resource/input/semantic/entity2id.txt")
    writeOutput([strKey + ' ' + str(dicRel[strKey]) for strKey in dicRel.keys()],
                "../resource/input/semantic/relation2id.txt")
    writeOutput(sorted(list(setRel)), "../resource/input/semantic/whole.txt")


def test():
    from tensorflow.python import pywrap_tensorflow
    chkp = "/TM/jaeh/RemoteInterpreter/contextualization/UMLS2vec/result/semantic/checkpoints/semantic-200"
    # chkp = "/TM/jaeh/RemoteInterpreter/contextualization/UMLS2vec/result/concept_partial/checkpoints/concept_partial-200"

    reader = pywrap_tensorflow.NewCheckpointReader(chkp)
    # print(reader.get_tensor('embedding/W').shape[0])
    np.save(chkp + ".npy", reader.get_tensor('embedding/W'))
    np.savetxt(chkp + ".txt", reader.get_tensor('embedding/W'))


def concept_partial():
    # strModel = "../result/semantic/checkpoints/semantic"
    strModel = "../result/concept_partial/checkpoints/concept_partial"
    # vec = torch.from_numpy(np.load(strModel + ".npy"))
    vec = np.load(strModel + ".npy")

    vocab = {}
    intIndex = 0
    with open(strModel + ".vocab") as f:
        for strVocab in f.read().split():
            vocab[strVocab] = intIndex
            intIndex += 1
    print(len(vocab))
    print(vec.shape)
    return vocab, vec

if __name__ == '__main__':
    # read_data_semantic()
    # read_data_concept()
    # read_data_concept_sourceFIltered()
    split_data()
    # concept_partial()
    # resource_check()
    measure()