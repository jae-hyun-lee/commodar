import time
from model import MGNC_CNN
import utilities

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import copy
from sklearn.metrics import precision_recall_fscore_support

init_time = time.time()

def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+')
    strNewRow = '\n'.join(listString)
    manipulatedData.write(strNewRow)
    manipulatedData.close()


def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


def npy_load(strModel):
    vec = np.load(strModel + ".npy")

    vocab = {}
    intIndex = 0
    with open(strModel + ".vocab") as f:
        for strVocab in f.read().split():
            vocab[strVocab] = intIndex
            intIndex += 1

    return vocab, vec


def train(data, params, pretrained=False, boolShuffle=True):
    boolNgram = False
    boolDependency = False
    boolKnowledge = False

    if params["MODALITY"] in set(["N", "ND", "NK", "NDK"]):
        boolNgram = True
    if params["MODALITY"] in set(["D", "ND", "DK", "NDK"]):
        boolDependency = True
    if params["MODALITY"] in set(["NK", "DK", "NDK"]):
        boolKnowledge = True

    if pretrained:
        params["WV_MATRIX"] = np.zeros((params["VOCAB_SIZE"] + 2, params["WORD_DIM"]))
        params["WVF_MATRIX"] = np.zeros((params["VOCAB_SIZE"] + 2, params["WORD_DIM"]))
        params["CONCEPT_MATRIX"] = np.zeros((params["CONCEPT_SIZE"] + 1, params["CONCEPT_DIM"]))
        model = utilities.load_model(MGNC_CNN(**params), params)
        model.train()

    else:
        # load word2vec
        if boolNgram:
            word_vectors = KeyedVectors.load_word2vec_format("n-gram.model.bin", binary=True)
            wv_matrix = []

        if boolDependency:
            word_dep_vocab, word_dep_vectors = npy_load("dependency")
            wvf_matrix = []

        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if boolNgram:
                if word in word_vectors.vocab:
                    wv_matrix.append(word_vectors.word_vec(word))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))

            if boolDependency:
                if word in word_dep_vocab:
                    wvf_matrix.append(word_dep_vectors[word_dep_vocab[word]])
                else:
                    wvf_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))

        if boolNgram:
            # one for unknown and one for zero padding
            wv_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))
            wv_matrix.append(np.zeros(params["WORD_DIM"]).astype("float16"))
            wv_matrix = np.array(wv_matrix)
            params["WV_MATRIX"] = wv_matrix

        if boolDependency:
            wvf_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))
            wvf_matrix.append(np.zeros(params["WORD_DIM"]).astype("float16"))
            wvf_matrix = np.array(wvf_matrix)
            params["WVF_MATRIX"] = wvf_matrix

        concept_matrix = []
        if boolKnowledge:
            concept_vocab, concept_vectors = npy_load("knowledge_triplet_" + params["DATASET"].replace("context", ""))

            for concept in data["concept"]:
                if concept in concept_vocab:
                    concept_matrix.append(concept_vectors[concept_vocab[concept]])
                else:
                    concept_matrix.append(np.zeros(params["CONCEPT_DIM"]).astype("float16"))

            # one for zero padding
            concept_matrix.append(np.zeros(params["CONCEPT_DIM"]).astype("float16"))
            concept_matrix = np.array(concept_matrix)
            params["CONCEPT_MATRIX"] = concept_matrix

        model = MGNC_CNN(**params)

    model = model.to('cuda')
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if params["OPTIMIZATION"] == "adadelta":
        optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    elif params["OPTIMIZATION"] == "adam":
        optimizer = optim.Adam(parameters, params["LEARNING_RATE"])
    elif params["OPTIMIZATION"] == "adamax":
        optimizer = optim.Adamax(parameters, params["LEARNING_RATE"])
    elif params["OPTIMIZATION"] == "sgd":
        optimizer = optim.SGD(parameters, params["LEARNING_RATE"], momentum=0.9)
    elif params["OPTIMIZATION"] == "adagrad":
        optimizer = optim.Adagrad(parameters, params["LEARNING_RATE"])
    elif params["OPTIMIZATION"] == "sparseadam":
        optimizer = optim.SparseAdam(parameters, params["LEARNING_RATE"])
    elif params["OPTIMIZATION"] == "RMSprop":
        optimizer = optim.RMSprop(parameters, params["LEARNING_RATE"])

    criterion = nn.CrossEntropyLoss()

    pre_dev_fsc = 0
    max_dev_fsc = 0

    listLog = []
    int_drop_sample_size = len(data["train_sen"]) % params["BATCH_SIZE"]
    for e in range(params["EPOCH"]):
        data2 = {}
        if boolShuffle:
            if boolKnowledge:
                data["train_sen"], data["train_class"], data["train_concept"] = shuffle(data["train_sen"], data["train_class"], data["train_concept"])
                data2["train_sen"], data2["train_class"], data2["train_concept"] = data["train_sen"][int_drop_sample_size:], data["train_class"][int_drop_sample_size:], data["train_concept"][int_drop_sample_size:]

            else:
                data["train_sen"], data["train_class"] = shuffle(data["train_sen"], data["train_class"])
                data2["train_sen"], data2["train_class"] = data["train_sen"][int_drop_sample_size:], data["train_class"][int_drop_sample_size:]

        for i in range(0, len(data2["train_sen"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data2["train_sen"]) - i)

            batch_sen = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data2["train_sen"][i:i + batch_range]]

            batch_class = [data["classes"].index(c) for c in data2["train_class"][i:i + batch_range]]
            batch_sen = Variable(torch.LongTensor(batch_sen)).cuda(params["GPU"])
            batch_class = Variable(torch.LongTensor(batch_class)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            if boolKnowledge:
                batch_concept = [[data["concept_to_idx"][concept] if concept in data["concept"] else params["CONCEPT_SIZE"] for concept in seq] +
                                 [params["CONCEPT_SIZE"]] * (params["MAX_CONCEPT_LEN"] - len(seq))
                                 for seq in data2["train_concept"][i:i + batch_range]]
                batch_concept = Variable(torch.LongTensor(batch_concept)).cuda(params["GPU"])
                pred = model([batch_sen, batch_concept])
            else:
                pred = model([batch_sen])
            loss = criterion(pred, batch_class)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        tup_dev_fsc_micro, tup_dev_fsc_each = test(data, model, params, boolKnowledge, mode="dev")
        dev_fsc = tup_dev_fsc_micro[2]

        if params["EARLY_STOPPING"] and dev_fsc <= pre_dev_fsc:
            break
        else:
            pre_dev_fsc = dev_fsc

        if dev_fsc > max_dev_fsc:
            max_dev_fsc = dev_fsc
            tup_max_dev_fsc = tup_dev_fsc_micro
            best_model = copy.deepcopy(model)

    return best_model, tup_max_dev_fsc[2], tup_dev_fsc_each, listLog


def test(data, model, params, boolKnowledge, mode="test"):
    model.eval()

    if mode == "dev":
        test_sen, test_class, test_concept = data["dev_sen"] + data["test_sen"], data["dev_class"] + data["test_class"], data["dev_concept"] + data["test_concept"]
    elif mode == "test":
        test_sen, test_class, test_concept = data["test_sen"], data["test_class"], data["test_concept"]

    test_sen = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent]
                + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)) for sent in test_sen]

    test_sen = Variable(torch.LongTensor(test_sen)).cuda(params["GPU"])
    test_class = [data["classes"].index(c) for c in test_class]

    if boolKnowledge:
        test_concept = [[data["concept_to_idx"][concept] if concept in data["concept"] else params["CONCEPT_SIZE"]
                         for concept in seq] + [params["CONCEPT_SIZE"]] * (params["MAX_CONCEPT_LEN"] - len(seq))
                        for seq in test_concept]
        test_concept = Variable(torch.LongTensor(test_concept)).cuda(params["GPU"])

        pred = np.argmax(model([test_sen, test_concept]).cpu().data.numpy(), axis=1)
    else:
        pred = np.argmax(model([test_sen]).cpu().data.numpy(), axis=1)

    result_micro = precision_recall_fscore_support(test_class, pred, average='micro')
    result_each = precision_recall_fscore_support(test_class, pred)
    return result_micro, result_each


def main():
    dataset = "context_finetuned"
    # dataset = "context_pretrained"
    data = getattr(utilities, f"read_{dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"] for w in sent])))
    data["concept"] = sorted(list(set([concept for seq in data["train_concept"] + data["dev_concept"] + data["test_concept"] for concept in seq])))
    data["classes"] = sorted(list(set(data["train_class"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    data["concept_to_idx"] = {w: i for i, w in enumerate(data["concept"])}
    data["idx_to_concept"] = {i: w for i, w in enumerate(data["concept"])}
    MAX_SENT_LEN = max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]])
    listFilter = [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23], [27, 28, 29]]
    FILTER_NUM = [100, 100, 100]
    FILTER_PRE = 3
    FILTER_NUM_PRE = 100
    listModality = ["N", "D", "ND", "NK", "DK", "NDK"]
    for modality in listModality:
        for FILTERS in listFilter:
            params = {
                "MODALITY": modality,
                "DATASET": dataset,
                "SAVE_MODEL": True,
                "EARLY_STOPPING": True,
                "EPOCH": 500,
                "LEARNING_RATE": 0.001,
                "MAX_SENT_LEN": MAX_SENT_LEN,
                "BATCH_SIZE": 50,
                "WORD_DIM": 200,
                "VOCAB_SIZE": len(data["vocab"]),
                "CLASS_SIZE": len(data["classes"]),
                "CONCEPT_SIZE": len(data["concept"]),
                "FILTER_PRE": FILTER_PRE,
                "FILTER_NUM_PRE": FILTER_NUM_PRE,
                "FILTERS": FILTERS,
                "FILTER_NUM": FILTER_NUM,
                "DROPOUT_PROB": 0.5,
                "NORM_LIMIT": 3,
                "OPTIMIZATION": "adam"
            }
            model, dev_acc, dev_acc_each, listLog = train(data, params)
            if params["SAVE_MODEL"]:
                utilities.save_model(model, params)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    measure()
