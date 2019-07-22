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


def token_index(sentence, listPosition):
    strEntity = sentence[listPosition[0]:listPosition[1]]
    strPrequel = sentence[:listPosition[0]] + "EOS"
    intEntityLength = len(strEntity.split())
    intEntityStart = len(strPrequel.split())
    return list(range(intEntityStart - 1, intEntityStart + intEntityLength - 1))


def load_model(model, params):
    filter_size = ",".join([str(size) for size in params['FILTERS']])
    path = f"../../result/saved_models(fn50)/context_finetuned_MGNC_semantic_{params['MODALITY']}_0.001_{filter_size}.pt"
    pretrained_weights = torch.load(path)
    state = model.state_dict()
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


def load_finetuned_model(model, params):
    filter_size = ",".join([str(size) for size in params['FILTERS']])
    path = f"../../learning/result/saved_models(fn50)/context_finetuned_MGNC_semantic_NDK_0.001_{filter_size}.pt"
    pretrained_weights = torch.load(path)
    state = model.state_dict()
    state['embedding1.weight'] = pretrained_weights['embedding1.weight']
    for i in range(len(params["FILTERS"])):
        state[f'conv_n_{i}.weight'] = pretrained_weights[f'conv_n_{i}.weight']
        state[f'conv_n_{i}.bias'] = pretrained_weights[f'conv_n_{i}.bias']
        state[f'bn_n_{i}.weight'] = pretrained_weights[f'bn_n_{i}.weight']
        state[f'bn_n_{i}.bias'] = pretrained_weights[f'bn_n_{i}.bias']
        state[f'bn_n_{i}.running_mean'] = pretrained_weights[f'bn_n_{i}.running_mean']
        state[f'bn_n_{i}.running_var'] = pretrained_weights[f'bn_n_{i}.running_var']
        state[f'bn_n_{i}.num_batches_tracked'] = pretrained_weights[f'bn_n_{i}.num_batches_tracked']

    state['embedding2.weight'] = pretrained_weights['embedding2.weight']
    for i in range(len(params["FILTERS"])):
        state[f'conv_d_{i}.weight'] = pretrained_weights[f'conv_d_{i}.weight']
        state[f'conv_d_{i}.bias'] = pretrained_weights[f'conv_d_{i}.bias']
        state[f'bn_d_{i}.weight'] = pretrained_weights[f'bn_d_{i}.weight']
        state[f'bn_d_{i}.bias'] = pretrained_weights[f'bn_d_{i}.bias']
        state[f'bn_d_{i}.running_mean'] = pretrained_weights[f'bn_d_{i}.running_mean']
        state[f'bn_d_{i}.running_var'] = pretrained_weights[f'bn_d_{i}.running_var']
        state[f'bn_d_{i}.num_batches_tracked'] = pretrained_weights[f'bn_d_{i}.num_batches_tracked']

    state['embedding3.weight'] = pretrained_weights['embedding3.weight']
    state[f'conv_k_0.weight'] = pretrained_weights[f'conv_k_0.weight']
    state[f'conv_k_0.bias'] = pretrained_weights[f'conv_k_0.bias']
    state[f'bn_k_0.weight'] = pretrained_weights[f'bn_k_0.weight']
    state[f'bn_k_0.bias'] = pretrained_weights[f'bn_k_0.bias']
    state[f'bn_k_0.running_mean'] = pretrained_weights[f'bn_k_0.running_mean']
    state[f'bn_k_0.running_var'] = pretrained_weights[f'bn_k_0.running_var']
    state[f'bn_k_0.num_batches_tracked'] = pretrained_weights[f'bn_k_0.num_batches_tracked']

    model.load_state_dict(state)
    return model


def load_dictionary(key):
    dic = {}
    path = f"../../learning/result/saved_dictionary/context_{key}.tsv"
    with open(path, "r") as fileInput:
        for strInstance in fileInput:
            listInstance = stringCleansing(strInstance).split("\t")
            if "idx_to_" in key:
                dic[int(listInstance[0])] = listInstance[1]
            else:
                dic[listInstance[0]] = int(listInstance[1])
    return dic


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


def read_context_unlabeled(strFile):
    data = {"sen": [], "triplet": [], "id": []}
    with open(strFile, "r", encoding="utf-8") as fileInput:
        for strInstance in fileInput:
            strInstance = stringCleansing(strInstance)
            listInstance = strInstance.split("\t")

            # instance integrity check
            if len(listInstance) < 21:
                continue

            # entity overlap
            if (listInstance[5].lower() == listInstance[11].lower())\
                    | (listInstance[11].lower() == listInstance[17].lower())\
                    | (listInstance[17].lower() == listInstance[5].lower()):
                continue

            if listInstance[7] == listInstance[13]:
                continue

            # entity normalization failure
            if (listInstance[20] == "NA") | (listInstance[8] == "NA") | (listInstance[14] == "NA"):
                continue

            listTokenConcept = [listInstance[20], listInstance[8], listInstance[14]]

            strToken, listCharPosition = clean_str(listInstance[3], [listInstance[4], listInstance[10], listInstance[16]])
            listToken = strToken.split()

            listEntity1Index = token_index(strToken, listCharPosition[:2])
            listEntity2Index = token_index(strToken, listCharPosition[2:4])
            listContextIndex = token_index(strToken, listCharPosition[4:])

            # entity span overlap
            if len(set(listEntity1Index).intersection(set(listEntity2Index)).union(
                    set(listEntity2Index).intersection(set(listContextIndex)),
                    set(listEntity1Index).intersection(set(listContextIndex)))) > 0:
                continue

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

            data["sen"].append(listToken)
            data["triplet"].append(listTokenConcept)
            data["id"].append(strInstance)

    return data