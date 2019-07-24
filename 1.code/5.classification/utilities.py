from sklearn.utils import shuffle
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
    entity1Pos = pos[0]
    entity2Pos = pos[1]
    listPosition = [int(i) for i in entity1Pos.split("-")] + [int(i) for i in entity2Pos.split("-")]

    if len(pos) > 2:
        contextPos = pos[2]
        listPosition += [int(i) for i in contextPos.split("-")]

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


def token_index(sentence, listPosition):
    strEntity = sentence[listPosition[0]:listPosition[1]]
    strPrequel = sentence[:listPosition[0]] + "EOS"
    intEntityLength = len(strEntity.split())
    intEntityStart = len(strPrequel.split())
    return list(range(intEntityStart - 1, intEntityStart + intEntityLength - 1))


def read_context_finetuned(boolShuffle=True):
    boolPPI = False
    undersampling_rate = 1
    boolNegativeSampling = False
    boolReverse = True

    boolSaveSet = False
    intSemantic = 0

    data = {}
    dicData = {}
    setConcept = set()

    with open("fine-tuningtsv", "r", encoding="utf-8") as fileInput:
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
                    strLabel = "NA|NA"
                else:
                    strLabel = '|'.join(listInstance[22:24])

            if boolSaveSet:
                setConcept = setConcept.union(set([listInstance[7 + intSemantic], listInstance[13 + intSemantic], listInstance[19 + intSemantic]]))

            if strLabel not in dicData:
                dicData[strLabel] = [[], [], [], 0]

            listTokenConcept = [listInstance[20], listInstance[8], listInstance[14]]

            # class: [ instance, concept, sentence ID, instance count in a class ]
            dicData[strLabel][0].append(listToken)
            dicData[strLabel][1].append(listTokenConcept)
            dicData[strLabel][2].append(listInstance[2])
            dicData[strLabel][3] += 1

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
        manipulatedData = open("knowledge_fine-tuning.vocab", 'w+')
        strNewRow = ' '.join(list(setConcept))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concepts: ", len(setConcept))

        concept_semantic("fine-tuning")

    return data


def read_context_pretrained(boolShuffle=True):
    boolSaveSet = False
    boolOversampling = False
    oversampling_rate = 1
    undersampling_rate = 1
    boolNegativeSampling = True
    intSemantic = 1

    data = {}
    dicData = {}
    setConceptComposition = set()

    listFile = ["pre-training.tsv", "pre-training_EVEX.tsv"]
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
                    continue

                # molecule entity 2 position indecator
                listToken.insert(listEntity2Index[-1] + 1, "</e2>")
                listToken.insert(listEntity2Index[0], "<e2>")

                # molecule entity 1 position indecator
                listToken.insert(listEntity1Index[-1] + 1, "</e1>")
                listToken.insert(listEntity1Index[0], "<e1>")

                # [ relation type, direction ]
                strLabel = "|".join(listInstance[16:])

                # concept ID @ '|'.join(list of semantic types)
                # [c, e1, e2]
                if boolSaveSet:
                    setConceptComposition = setConceptComposition.union(set([listInstance[7] + "@" + listInstance[8], listInstance[13] + "@" + listInstance[14]]))

                listTokenConcept = ["dummy", listInstance[8], listInstance[14]]

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


    if boolSaveSet:
        manipulatedData = open("knowledge_pre-training.vocab", 'w+')

        setEntity = set([strComposition.split("@")[intSemantic] for strComposition in setConceptComposition])
        strNewRow = ' '.join(list(setEntity))
        manipulatedData.write(strNewRow)
        manipulatedData.close()
        print("# of concept compositions: ", len(setConceptComposition))
        listName = ["concept", "semantic type"]
        print(f"# of {listName[intSemantic]}: ", len(setEntity))
        concept_semantic("pre-training")

    return data


def save_model(model, params):
    filter_size = ",".join([str(size) for size in params['FILTERS']])
    path = f"classification_{params['DATASET']}_{params['MODALITY']}_{filter_size}.pt"
    torch.save(model.module.state_dict(), path)


def save_dictionary(dic, key):
    path = f"context_{key}.tsv"
    writeOutput([str(strKey) + "\t" + str(dic[strKey]) for strKey in dic.keys()], path)


def load_dictionary(key):
    dic = {}
    path = f"context_{key}.tsv"
    with open(path, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringCleansing(strInstance).split("\t")
            if "idx_to_" in key:
                dic[int(listInstance[0])] = listInstance[1]
            else:
                dic[listInstance[0]] = int(listInstance[1])
    return dic


def save_class(list, key):
    path = f"context_class_{key}.tsv"
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
    vec = np.load(strModel + ".npy")

    vocab = {}
    intIndex = 0
    with open(strModel + ".vocab") as f:
        for strVocab in f.read().split():
            vocab[strVocab] = intIndex
            intIndex += 1

    return vocab, vec


def concept_semantic(dataset):
    with open(dataset + ".vocab", "r", encoding="utf-8") as fileInput:
        listSemantic = fileInput.readline().split()

    semantic_vocab, semantic_vectors = npy_load("knowledge_triplet")
    concept_matrix = []

    for strSemantic in listSemantic:
        if strSemantic in semantic_vocab:
            concept_matrix.append(np.mean([semantic_vectors[semantic_vocab[strType]] for strType in strSemantic.split('|')], axis=0))
        else:
            concept_matrix.append(np.random.uniform(-0.01, 0.01, 10).astype("float16"))

    np.save("knowledge_triplet_" + dataset + ".npy", concept_matrix)
