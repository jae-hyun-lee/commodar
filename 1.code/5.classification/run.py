import time
# from learning.code.galsang.model import CNN
# from model import CNN_shallow
from model import MGNC_CNN_shallow
from model import depthwiseCNN
import utilities

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
from sklearn.metrics import precision_recall_fscore_support

# import xml.etree.ElementTree as ET

init_time = time.time()

def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+');
    strNewRow = '\n'.join(listString);
    manipulatedData.write(strNewRow);
    manipulatedData.close();


def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


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


def train(data, params, pretrained=False):
    if params["MODEL"] != "rand":
        # load word2vec
        # print("loading representation models...")

        word_vectors = KeyedVectors.load_word2vec_format("../../../word2vec/result/test.model.bin", binary=True)
        # word_vectors = KeyedVectors.load_word2vec_format("../../../word2vec/result/word2vec_whole.model.bin", binary=True)
        wv_matrix = []

        word_dep_vocab, word_dep_vectors = npy_load("../../../word2vec/result/word2vecf_200_min5_np_includingGE,PC")
        wvf_matrix = []

        concept_vocab, concept_vectors = npy_load("../../resource/concept_composition" + params["DATASET"].replace("context", ""))
        concept_matrix = []

        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float32"))
                # print(word, "not in wv")

            if word in word_dep_vocab:
                wvf_matrix.append(word_dep_vectors[word_dep_vocab[word]])
            else:
                wvf_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float32"))
                    # print(word, "not in wvf")

        for concept in data["concept"]:
            if concept in concept_vocab:
                concept_matrix.append(concept_vectors[concept_vocab[concept]])
            else:
                concept_matrix.append(np.zeros(params["WORD_DIM"]).astype("float32"))

        # one for unknown and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float32"))
        wv_matrix.append(np.zeros(params["WORD_DIM"]).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

        wvf_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float32"))
        wvf_matrix.append(np.zeros(params["WORD_DIM"]).astype("float32"))
        wvf_matrix = np.array(wvf_matrix)
        params["WVF_MATRIX"] = wvf_matrix

        # one for zero padding
        concept_matrix.append(np.zeros(params["WORD_DIM"]).astype("float32"))
        concept_matrix = np.array(concept_matrix)
        params["CONCEPT_MATRIX"] = concept_matrix

    if pretrained:
        model = utilities.load_model(CNN_shallow(**params), params)
        model.train()
    else:
        model = CNN_shallow(**params)
        # model = CNN(**params).cuda(params["GPU"])

    model = model.to('cuda')
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # print(model)
    # print(params["CLASS_SIZE"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = optim.Adam(parameters, params["LEARNING_RATE"])
    # optimizer = optim.Adamax(parameters, params["LEARNING_RATE"])
    # optimizer = optim.ASGD(parameters, params["LEARNING_RATE"])

    criterion = nn.CrossEntropyLoss()

    pre_dev_fsc = 0
    max_dev_fsc = 0

    for e in range(params["EPOCH"]):
        data["train_sen"], data["train_class"], data["train_concept"] = shuffle(data["train_sen"], data["train_class"], data["train_concept"])

        for i in range(0, len(data["train_sen"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_sen"]) - i)

            batch_sen = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_sen"][i:i + batch_range]]

            batch_class = [data["classes"].index(c) for c in data["train_class"][i:i + batch_range]]

            batch_concept = [[data["concept_to_idx"][concept] for concept in seq] +
                         [params["CONCEPT_SIZE"]] * (params["MAX_SENT_LEN"] - len(seq))
                         for seq in data["train_concept"][i:i + batch_range]]

            batch_sen = Variable(torch.LongTensor(batch_sen)).cuda(params["GPU"])
            batch_class = Variable(torch.LongTensor(batch_class)).cuda(params["GPU"])
            batch_concept = Variable(torch.LongTensor(batch_concept)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model([batch_sen, batch_concept])
            loss = criterion(pred, batch_class)
            loss.backward()
            # print(loss)

            torch.nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

            # dicClass = {}
            # for i in range(len(data["train_class"])):
            #     if data["train_class"][i] in dicClass:
            #         dicClass[data["train_class"][i]] += 1
            #     else:
            #         dicClass[data["train_class"][i]] = 1
            # print(dicClass)

        tup_dev_fsc, tup_dev_fsc_each = test(data, model, params, mode="dev")
        dev_fsc = tup_dev_fsc[2]
        # test_fsc = test(data, model, params)
        # print("epoch:", e + 1, "/ dev_fsc:", dev_fsc, "/ test_fsc:", test_fsc)

        if params["EARLY_STOPPING"] and dev_fsc <= pre_dev_fsc:
            print("early stopping by dev_fsc!")
            break
        else:
            pre_dev_fsc = dev_fsc

        if dev_fsc > max_dev_fsc:
            max_dev_fsc = dev_fsc
            tup_max_dev_fsc = tup_dev_fsc
            # max_test_fsc = test_fsc
            best_model = copy.deepcopy(model)

        if e % 100 == 0:
            print(tup_dev_fsc_each)
    # print("max dev fsc:", max_dev_fsc, "test fsc:", max_test_fsc)
    # return best_model, max_dev_fsc, max_test_fsc

    return best_model, '/'.join("{:.2f}".format(float(metric)) for metric in tup_max_dev_fsc[:3]), tup_dev_fsc_each


def train_ONOFF(data, params, pretrained=False, boolShuffle=True):
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
        model = utilities.load_model(depthwiseCNN(**params), params) if params["DEPTH"] else utilities.load_model(MGNC_CNN_shallow(**params), params)
        model.train()

    else:
        # load word2vec
        # print("loading representation models...")
        if boolNgram:
            # word_vectors = KeyedVectors.load_word2vec_format("../../../word2vec/result/test.model.bin", binary=True)
            word_vectors = KeyedVectors.load_word2vec_format("../../../word2vec/result/word2vec_whole.model.bin", binary=True)
            wv_matrix = []

        if boolDependency:
            word_dep_vocab, word_dep_vectors = npy_load("../../../word2vecf/result/word2vecf_200_min5_np_2015")
            wvf_matrix = []

        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if boolNgram:
                if word in word_vectors.vocab:
                    wv_matrix.append(word_vectors.word_vec(word))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))
                    # print(word, "not in wv")

            if boolDependency:
                if word in word_dep_vocab:
                    wvf_matrix.append(word_dep_vectors[word_dep_vocab[word]])
                else:
                    wvf_matrix.append(np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16"))
                    # print(word, "not in wvf")

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
            if params["KNOWLEDGE"] == "semantic":
                concept_vocab, concept_vectors = npy_load("../../resource/concept_semantic" + params["DATASET"].replace("context", ""))
            else:
                concept_vocab, concept_vectors = npy_load("../../resource/concept" + params["DATASET"].replace("context", ""))

            for concept in data["concept"]:
                if concept in concept_vocab:
                    concept_matrix.append(concept_vectors[concept_vocab[concept]])
                else:
                    concept_matrix.append(np.zeros(params["CONCEPT_DIM"]).astype("float16"))

            # one for zero padding
            concept_matrix.append(np.zeros(params["CONCEPT_DIM"]).astype("float16"))
            concept_matrix = np.array(concept_matrix)
            params["CONCEPT_MATRIX"] = concept_matrix

        model = depthwiseCNN(**params)if params["DEPTH"] else MGNC_CNN_shallow(**params)

    model = model.to('cuda')
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # print(model)
    # print(params["CLASS_SIZE"])

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
    max_test_fsc= 0

    listLog = []
    int_drop_sample_size = len(data["train_sen"]) % params["BATCH_SIZE"]
    # print(int_drop_sample_size)

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
            # print(i)
            batch_range = min(params["BATCH_SIZE"], len(data2["train_sen"]) - i)

            batch_sen = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data2["train_sen"][i:i + batch_range]]

            batch_class = [data["classes"].index(c) for c in data2["train_class"][i:i + batch_range]]
            # batch_id = [data["id"].index(c) for c in data2["train_id"][i:i + batch_range]]

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
            # print(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        tup_dev_fsc_micro, tup_dev_fsc_each = test(data, model, params, boolKnowledge, mode="dev")
        dev_fsc = tup_dev_fsc_micro[2]
        # tup_test_fsc, tup_test_fsc_each = test(data, model, params)
        # print("epoch:", e + 1, "/ dev_fsc:", dev_fsc, "/ test_fsc:", test_fsc)

        if params["EARLY_STOPPING"] and dev_fsc <= pre_dev_fsc:
            # print("early stopping by dev_fsc!", e)
            break
        else:
            pre_dev_fsc = dev_fsc

        if dev_fsc > max_dev_fsc:
            max_dev_fsc = dev_fsc
            tup_max_dev_fsc = tup_dev_fsc_micro
            # tup_max_test_fsc = tup_test_fsc
            best_model = copy.deepcopy(model)

    #    if e % 100 == 0:
    #        listLine = []
    #        for classes in tup_dev_fsc_each[:3]:
    #            for score in classes:
    #                listLine.append("{:.3f}".format(float(score)))
    #            listLine.append("")
    #         print(tup_dev_fsc_each)
    #         print("\t".join(listLine))
    #        listLog.append("\t".join(listLine))
    # print("max dev fsc:", max_dev_fsc, "test fsc:", max_test_fsc)
    # return best_model, max_dev_fsc, max_test_fsc

    return best_model, tup_max_dev_fsc[2], tup_dev_fsc_each, listLog


def test(data, model, params, boolKnowledge, mode="test"):
    model.eval()

    if mode == "dev":
        # test_sen, test_class, test_concept, test_id = data["dev_sen"], data["dev_class"], data["dev_concept"], data["dev_id"]
        test_sen, test_class, test_concept = data["dev_sen"] + data["test_sen"], data["dev_class"] + data["test_class"], data["dev_concept"] + data["test_concept"]
    elif mode == "test":
        test_sen, test_class, test_concept = data["test_sen"], data["test_class"], data["test_concept"]

    # dicClass = {}
    # for i in range(len(data["dev_class"])):
    #     if data["dev_class"][i] in dicClass:
    #         dicClass[data["dev_class"][i]] += 1
    #     else:
    #         dicClass[data["dev_class"][i]] = 1
    # print(dicClass)

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
    # acc = sum([1 if p == y else 0 for p, y in zip(pred, test_class)]) / len(pred)

    result_micro = precision_recall_fscore_support(test_class, pred, average='micro')
    result_each = precision_recall_fscore_support(test_class, pred)
    return result_micro, result_each


def main_backup():
    dataset = "context_finetuned"
    # dataset = "context_pretrained"

    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="multichannel", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default=dataset, help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=500, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    data = getattr(utilities, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"] for w in sent])))
    data["concept"] = sorted(list(set([concept for seq in data["train_concept"] + data["dev_concept"] + data["test_concept"] for concept in seq])))
    data["classes"] = sorted(list(set(data["train_class"])))
    # print(len(data["train_class"]))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    data["concept_to_idx"] = {w: i for i, w in enumerate(data["concept"])}
    data["idx_to_concept"] = {i: w for i, w in enumerate(data["concept"])}

    MAX_SENT_LEN = max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]])
    # strTable = "../../result/log/log181203_PRF(adam).tsv"
    # strTable = "../../result/log/pretrained_log181203_PRF(adam).tsv"
    # strTable = "../../result/log/transfer_log190304_PRF(adam).tsv"
    strTable = "../../result/log/PI_shallow_log190415_PRF(adam)_NK.tsv"

    # listLearningRate = [0.1, 0.05, 0.01]
    # listLearningRate = [0.05, 0.01, 0.005, 0.001]
    listLearningRate = [0.1]

    listFilterPre = [2, 8, 14, 20]

    listFilter = [[3, 4, 5], [9, 10, 11],  [15, 16, 17], [3, 10, 17]]
    # listFilter = [[3, 4, 5]]

    # listBatch = [5, 10, 15, 20, 25, 50, 75, 100]
    listBatch = [50]

    FILTER_NUM_PRE = 100
    FILTER_NUM = [100, 100, 100]

    for learning_rate in listLearningRate:
        manipulatedData = open(strTable, 'a')
        manipulatedData.write("learning_rate\t" + str(learning_rate) + "\n")
        manipulatedData.write("filter/batch_size\t" + "\t".join([str(bat) for bat in listBatch]) + "\n")
        manipulatedData.close()
        # print("learning_rate\t" + str(learning_rate))
        # print("filter/batch_size\t" + "\t".join([str(bat) for bat in listBatch]))

        for FILTERS in listFilter:
            listWrite = []
            print(FILTERS)
            for BATCH_SIZE in listBatch:
                for FILTER_PRE in listFilterPre:
                    print(FILTER_PRE)

                    params = {
                        "MODEL": options.model,
                        "DATASET": dataset,
                        "SAVE_MODEL": options.save_model,
                        "EARLY_STOPPING": options.early_stopping,
                        "EPOCH": options.epoch,
                        "LEARNING_RATE": learning_rate,
                        "MAX_SENT_LEN": MAX_SENT_LEN,
                        # "MAX_SENT_LEN": 158,
                        "BATCH_SIZE": BATCH_SIZE,
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
                        "GPU": options.gpu
                    }
                    # print(max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]]))
                    # print("=" * 20 + "INFORMATION" + "=" * 20)
                    # print("MODEL:", params["MODEL"])
                    # print("DATASET:", params["DATASET"])
                    # print("VOCAB_SIZE:", params["VOCAB_SIZE"])
                    # print("EPOCH:", params["EPOCH"])
                    # print("LEARNING_RATE:", params["LEARNING_RATE"])
                    # print("EARLY_STOPPING:", params["EARLY_STOPPING"])
                    # print("SAVE_MODEL:", params["SAVE_MODEL"])
                    # print("=" * 20 + "INFORMATION" + "=" * 20)

                    if options.mode == "train":
                        # print("=" * 20 + "TRAINING STARTED" + "=" * 20)
                        # model, dev_acc, test_acc = train(data, params)
                        if "pretrained" in params["DATASET"]:
                            model, dev_acc, dev_acc_each = train(data, params)
                        elif "finetuned" in params["DATASET"]:
                            model, dev_acc, dev_acc_each = train(data, params, pretrained=True)

                        if params["SAVE_MODEL"]:
                            utilities.save_model(model, params)
                        # print("=" * 20 + "TRAINING FINISHED" + "=" * 20)

                    else:
                        model = utilities.load_model(params).cuda(params["GPU"])
                        test_acc = test(data, model, params)
                        print("test acc:", test_acc)

                    print(dev_acc_each)
                    torch.cuda.empty_cache()
                    listWrite.append(dev_acc)
                    # print(str(FILTER_PRE), dev_acc)
                    # measure()

                manipulatedData = open(strTable, 'a')
                manipulatedData.write(str(",".join([str(fil) for fil in FILTERS])) + "\t" + "\t".join(listWrite) + "\n")
                manipulatedData.close()
            # print(str(",".join([str(fil) for fil in FILTERS])) + "\t" + "\t".join(listWrite))
            measure()
        #
        # manipulatedData = open(strTable, 'a')
        # manipulatedData.write("\n")
        # manipulatedData.close()


def main_shallow():
    # listDataset = ["context_pretrained_MGNC", "context_finetuned_MGNC"]
    # listDataset = ["context_finetuned_MGNC"]
    listDataset = ["context_pretrained_MGNC"]

    # file_head = "../../result/log/performance_pretrained_es(False).tsv"
    # # file_head = "../../result/log/performance_10(True).tsv"
    # manipulatedData = open(file_head, 'a')
    for dataset in listDataset:
        iter = 1 if "pretrained" in dataset else 10
        file_head = "../../result/log/performance" + dataset.replace("context", "").replace("MGNC", "") + "es(False)190630.tsv"
        manipulatedData = open(file_head, 'a')
        for i in range(iter):
            print(i)
            # manipulatedData.write(str(i) + "\n")

            pretraining = True if "pretrained" in dataset else False
            # print(dataset)
            parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
            parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
            # parser.add_argument("--dataset", default=dataset, help="available datasets: MR, TREC")
            parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
            parser.add_argument("--early_stopping", default=True, action='store_true', help="whether to apply early stopping")
            parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
            # parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
            parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")

            options = parser.parse_args()
            data = getattr(utilities, f"read_{dataset}")(True)
            data["classes"] = sorted(list(set(data["train_class"])))
            # print(data["classes"])
            MAX_SENT_LEN = max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]]) if pretraining else 237

            listLearningRate = [0.001]
            MAX_CONCEPT_LEN = 3 if "MGNC" in dataset else MAX_SENT_LEN

            # listDepth = [True]
            listDepth = [False]
            for depth in listDepth:
                # print(f"depth: {depth}")
                # listFilterNumConcept = [10, 50, 100]
                listFilterNumConcept = [50, 100]

                for filter_num_concept in listFilterNumConcept:
                    manipulatedData.write(str(filter_num_concept) + "\n")
                    # print(filter_num_concept)
                    # file_head = "../../result/log/PRF_190605_" + dataset.replace("context_", "").replace("_MCNG", "")
                    # strTable = file_head + ".tsv"
                    # strLog = file_head + "_log.tsv"
                    listFilter = [[33, 34, 35]]
                    # listFilter = [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]]

                    if pretraining:
                        listBatch = [50]
                        data["vocab"] = sorted(list(set([w for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"] for w in sent])))
                        data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
                        data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
                        # utilities.save_dictionary(data["word_to_idx"], "word_to_idx")
                        # utilities.save_dictionary(data["idx_to_word"], "idx_to_word")
                        # utilities.save_class(data["classes"], "pretrained")

                    else:
                        # listBatch = [5, 25, 50, 75]
                        listBatch = [5]
                        data["word_to_idx"] = utilities.load_dictionary("word_to_idx_finetuned")
                        data["idx_to_word"] = utilities.load_dictionary("idx_to_word_finetuned")
                        data["vocab"] = sorted(data["word_to_idx"].keys())
                        # utilities.save_class(data["classes"], "finetuned")

                    # vocab_size = len(data["vocab"]) if pretraining else 55934
                    vocab_size = len(data["vocab"])
                    FILTER_NUM = [100, 100, 100]

                    # listModality = ["N", "D", "ND", "NK", "DK", "NDK"]
                    # listModality = ["N", "D", "ND", "NK", "DK", "NDK"] if filter_num_concept == 100 else ["NK", "DK", "NDK"]
                    listModality = ["N", "D", "ND"] if filter_num_concept == 100 else ["NK", "DK", "NDK"]
                    # listModality = ["NK", "DK", "NDK"]

                    # listOpt = ["adam", "adamax", "sgd", "adadelta", "adagrad", "sparseadam", "RMSprop"]
                    listOpt = ["adam"]

                    # listKnowledge = ["semantic", "concept"]
                    listKnowledge = ["semantic"]

                    listWrite = ["learning_rate", "batch_size", "filter_size"]
                    for learning_rate in listLearningRate:
                        listWrite[0] += "\t" + str(learning_rate) + "\t" * (len(listFilter) * len(listBatch) - 1)
                        for BATCH_SIZE in listBatch:
                            listWrite[1] += "\t" + str(BATCH_SIZE) + "\t" * (len(listFilter) - 1)
                            for FILTERS in listFilter:
                                listWrite[2] += "\t" + ",".join([str(fil) for fil in FILTERS])
                    # manipulatedData.write("\n".join(listWrite) + "\n")
                    listLogWrite = []

                    listMeta = []
                    listSemantic = []
                    for listSen in data["train_concept"] + data["dev_concept"] + data["test_concept"]:
                        senMeta = []
                        senSemantic = []
                        for concept_composition in listSen:
                            listComposition = concept_composition.split("@")
                            if len(listComposition) > 1:
                                senMeta.append(listComposition[0])
                                senSemantic.append(listComposition[1])
                            else:
                                senMeta.append(concept_composition)
                                senSemantic.append(concept_composition)
                        listMeta.append(senMeta)
                        listSemantic.append(senSemantic)

                    for knowledge in listKnowledge:
                        # print(knowledge)
                        listWrite.append(knowledge)
                        concept_dim = 10 if knowledge == "semantic" else 190
                        listConcept = listSemantic if knowledge == "semantic" else listMeta
                        data["train_concept"] = listConcept[:len(data["train_concept"])]
                        data["dev_concept"] = listConcept[len(data["train_concept"]):len(data["train_concept"])+len(data["dev_concept"])]
                        data["test_concept"] = listConcept[len(data["train_concept"])+len(data["dev_concept"]):]

                        if pretraining:
                            data["concept"] = sorted(list(set([concept for seq in data["train_concept"] + data["dev_concept"] + data["test_concept"] for concept in seq])))
                            concept_size = len(data["concept"])
                            data["concept_to_idx"] = {w: i for i, w in enumerate(data["concept"])}
                            data["idx_to_concept"] = {i: w for i, w in enumerate(data["concept"])}
                            # utilities.save_dictionary(data["concept_to_idx"], "concept_to_idx")
                            # utilities.save_dictionary(data["idx_to_concept"], "idx_to_concept")
                        else:
                            data["concept_to_idx"] = utilities.load_dictionary("concept_to_idx_finetuned")
                            data["idx_to_concept"] = utilities.load_dictionary("idx_to_concept_finetuned")
                            data["concept"] = sorted(data["concept_to_idx"].keys())
                            # concept_size = 103 if knowledge == "semantic" else 10596
                            concept_size = len(data["concept"])

                        for opt in listOpt:
                            # print(opt)
                            for modality in listModality:
                                listLogWrite.append(modality)
                                strWrite = modality
                                # print(modality)
                                for learning_rate in listLearningRate:
                                    listLogWrite.append(str(learning_rate))
                                    # print(learning_rate)
                                    for BATCH_SIZE in listBatch:
                                        listLogWrite.append(str(BATCH_SIZE))
                                        # print(BATCH_SIZE)
                                        for FILTERS in listFilter:
                                            # listLogWrite.append(",".join([str(fil) for fil in FILTERS]))
                                            print(FILTERS)
                                            params = {
                                                "MODALITY": modality,
                                                "DATASET": dataset,
                                                "SAVE_MODEL": options.save_model,
                                                "EARLY_STOPPING": options.early_stopping,
                                                "EPOCH": options.epoch,
                                                "LEARNING_RATE": learning_rate,
                                                "MAX_SENT_LEN": MAX_SENT_LEN,
                                                "MAX_CONCEPT_LEN": MAX_CONCEPT_LEN,
                                                # "MAX_SENT_LEN": 158,
                                                "BATCH_SIZE": BATCH_SIZE,
                                                "WORD_DIM": 200,
                                                "CONCEPT_DIM": concept_dim,
                                                "VOCAB_SIZE": vocab_size,
                                                "CLASS_SIZE": len(data["classes"]),
                                                "CONCEPT_SIZE": concept_size,
                                                "FILTERS": FILTERS,
                                                "FILTER_NUM": FILTER_NUM,
                                                "FILTER_NUM_CONCEPT": filter_num_concept,
                                                "DROPOUT_PROB": 0.5,
                                                "NORM_LIMIT": 3,
                                                "GPU": options.gpu,
                                                "OPTIMIZATION": opt,
                                                "KNOWLEDGE": knowledge,
                                                "DEPTH": depth
                                            }

                                            # print(max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]]))
                                            # print("=" * 20 + "INFORMATION" + "=" * 20)
                                            # print("MODEL:", params["MODEL"])
                                            # print("DATASET:", params["DATASET"])
                                            # print("VOCAB_SIZE:", params["VOCAB_SIZE"])
                                            # print("EPOCH:", params["EPOCH"])
                                            # print("LEARNING_RATE:", params["LEARNING_RATE"])
                                            # print("EARLY_STOPPING:", params["EARLY_STOPPING"])
                                            # print("SAVE_MODEL:", params["SAVE_MODEL"])
                                            # print("=" * 20 + "INFORMATION" + "=" * 20)

                                            if options.mode == "train":
                                                # if "pretrained" in params["DATASET"]:
                                                #     model, dev_acc, dev_acc_each = train(data, params)
                                                # elif "finetuned" in params["DATASET"]:
                                                #     model, dev_acc, dev_acc_each = train(data, params, pretrained=True)
                                                model, dev_acc, dev_acc_each, listLog = train_ONOFF(data, params, pretrained=not pretraining)

                                                if params["SAVE_MODEL"]:
                                                    utilities.save_model(model, params)

                                            else:
                                                model = utilities.load_model(params).cuda(params["GPU"])
                                                test_acc = test(data, model, params)
                                                print("test acc:", test_acc)

                                            torch.cuda.empty_cache()
                                            strWrite += "\t" + "{0:.2f}".format(dev_acc)
                                            listLogWrite += listLog

                                listWrite.append(strWrite)
                                manipulatedData.write(strWrite + "\n")
                                print(strWrite)
        manipulatedData.close()


def main():
    dataset = "context_finetuned"
    # dataset = "context_pretrained"

    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="N", help="available models: N, D, ND, NK, DK, NDK")
    parser.add_argument("--dataset", default=dataset, help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=500, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    data = getattr(utilities, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"] for w in sent])))
    data["concept"] = sorted(list(set([concept for seq in data["train_concept"] + data["dev_concept"] + data["test_concept"] for concept in seq])))
    data["classes"] = sorted(list(set(data["train_class"])))
    # print(len(data["train_class"]))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    data["concept_to_idx"] = {w: i for i, w in enumerate(data["concept"])}
    data["idx_to_concept"] = {i: w for i, w in enumerate(data["concept"])}

    MAX_SENT_LEN = max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]])

    # listOpt = ["adam", "sgd", "adamax", "adagrad", "sparseadam", "RMSprop"]
    listOpt = ["adam", "sgd"]

    for opt in listOpt:
        print(opt)
        file_head = "../../result/log/PRF_190428_"
        # strTable = "../../result/log/log181203_PRF(adam).tsv"
        # strTable = "../../result/log/pretrained_log181203_PRF(adam).tsv"
        # strTable = "../../result/log/transfer_log190304_PRF(adam).tsv"
        strTable = file_head + opt + ".tsv"
        strLog = file_head + opt + "_log.tsv"
        manipulatedData = open(strTable, 'w')

        # listLearningRate = [0.01, 0.001]
        listLearningRate = [0.01]

        # listFilter = [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23], [27, 28, 29], [3, 16, 29]]
        listFilter = [[3, 4, 5], [27, 28, 29], [3, 16, 29]]

        listBatch = [5, 25, 50, 75, 100]
        # listBatch = [25]
        FILTER_NUM = [100, 100, 100]

        FILTER_PRE = 3
        FILTER_NUM_PRE = 100

        listWrite = ["learning_rate", "batch_size", "filter_size"]
        for learning_rate in listLearningRate:
            listWrite[0] += "\t" + str(learning_rate) + "\t"*(len(listFilter)*len(listBatch)-1)
            for BATCH_SIZE in listBatch:
                listWrite[1] += "\t" + str(BATCH_SIZE) + "\t"*(len(listFilter)-1)
                for FILTERS in listFilter:
                    listWrite[2] += "\t" + ",".join([str(fil) for fil in FILTERS])

        manipulatedData.write("\n".join(listWrite) + "\n")
        listLogWrite = []

        listModality = ["N", "D", "ND", "NK", "DK", "NDK"]

        for modality in listModality:
            listLogWrite.append(modality)
            strWrite = modality
            print(modality)
            for learning_rate in listLearningRate:
                listLogWrite.append(str(learning_rate))
                print(learning_rate)
                for BATCH_SIZE in listBatch:
                    listLogWrite.append(str(BATCH_SIZE))
                    print(BATCH_SIZE)
                    for FILTERS in listFilter:
                        listLogWrite.append(",".join([str(fil) for fil in FILTERS]))
                        print(",".join([str(fil) for fil in FILTERS]))
                        params = {
                            "MODALITY": modality,
                            "DATASET": dataset,
                            "SAVE_MODEL": options.save_model,
                            "EARLY_STOPPING": options.early_stopping,
                            "EPOCH": options.epoch,
                            "LEARNING_RATE": learning_rate,
                            "MAX_SENT_LEN": MAX_SENT_LEN,
                            # "MAX_SENT_LEN": 158,
                            "BATCH_SIZE": BATCH_SIZE,
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
                            "GPU": options.gpu,
                            "OPTIMIZATION": opt
                        }

                        # print(max([len(sent) for sent in data["train_sen"] + data["dev_sen"] + data["test_sen"]]))
                        # print("=" * 20 + "INFORMATION" + "=" * 20)
                        # print("MODEL:", params["MODEL"])
                        # print("DATASET:", params["DATASET"])
                        # print("VOCAB_SIZE:", params["VOCAB_SIZE"])
                        # print("EPOCH:", params["EPOCH"])
                        # print("LEARNING_RATE:", params["LEARNING_RATE"])
                        # print("EARLY_STOPPING:", params["EARLY_STOPPING"])
                        # print("SAVE_MODEL:", params["SAVE_MODEL"])
                        # print("=" * 20 + "INFORMATION" + "=" * 20)

                        if options.mode == "train":
                            # if "pretrained" in params["DATASET"]:
                            #     model, dev_acc, dev_acc_each = train(data, params)
                            # elif "finetuned" in params["DATASET"]:
                            #     model, dev_acc, dev_acc_each = train(data, params, pretrained=True)
                            model, dev_acc, dev_acc_each, listLog = train_ONOFF(data, params)

                            if params["SAVE_MODEL"]:
                                utilities.save_model(model, params)
                            # print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
                        else:
                            model = utilities.load_model(params).cuda(params["GPU"])
                            test_acc = test(data, model, params)
                            print("test acc:", test_acc)

                        torch.cuda.empty_cache()
                        strWrite += "\t" + dev_acc
                        listLogWrite += listLog

            # listWrite.append(strWrite)
            manipulatedData.write(strWrite + "\n")
        manipulatedData.close()

        # writeOutput(listWrite, strTable)
        writeOutput(listLogWrite, strLog)


def fold():
    fold = 5
    file_head = f"../../result/log/performance_{fold}fold_190630.tsv"
    manipulatedData = open(file_head, 'a')

    listModality = ["N", "D", "ND", "NK", "DK", "NDK"]
    listFilter = [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23], [27, 28, 29]]
    listFilter = [[33, 34, 35]]
    # manipulatedData.write("\t" + "\t".join([",".join([str(size) for size in filterSet]) for filterSet in listFilter]) + "\n")

    for i in range(10):
        print(i)
        loaded_data = utilities.read_context_finetuned_10fold(True)

        splitted_sentence = [[]] * fold
        splitted_concept = [[]] * fold
        splitted_class = [[]] * fold

        for key in loaded_data.keys():
            interval = round(len(loaded_data[key][0])/fold)
            sentences = [loaded_data[key][0][i*interval:(i+1)*interval] for i in range(fold-1)] + [loaded_data[key][0][(fold - 1) * interval:]]

            concepts = [loaded_data[key][1][i * interval:(i + 1) * interval] for i in range(fold-1)] + [loaded_data[key][1][(fold - 1) * interval:]]

            dummy_class = [key] * len(loaded_data[key][0])
            classes = [dummy_class[i * interval:(i + 1) * interval] for i in range(fold - 1)] + [dummy_class[(fold - 1) * interval:]]

            for i in range(fold):
                splitted_sentence[i] = splitted_sentence[i] + sentences[i]
                splitted_concept[i] = splitted_concept[i] + concepts[i]
                splitted_class[i] = splitted_class[i] + classes[i]

        data = {"test_sen": [], "test_class": [], "test_concept": []}
        data["classes"] = sorted(loaded_data.keys())
        data["word_to_idx"] = utilities.load_dictionary("word_to_idx_finetuned")
        data["idx_to_word"] = utilities.load_dictionary("idx_to_word_finetuned")
        data["vocab"] = sorted(data["word_to_idx"].keys())

        data["concept_to_idx"] = utilities.load_dictionary("concept_to_idx_finetuned")
        data["idx_to_concept"] = utilities.load_dictionary("idx_to_concept_finetuned")
        data["concept"] = sorted(data["concept_to_idx"].keys())

        for modality in listModality:
            strWrite = modality
            for FILTERS in listFilter:
                average = 0
                # print(FILTERS)
                for i in range(fold):
                    # print(i)
                    data["train_sen"], data["train_class"], data["train_concept"] = [], [], []
                    data["dev_sen"], data["dev_class"], data["dev_concept"] = splitted_sentence[i], splitted_class[i], splitted_concept[i]
                    for j in set(range(fold)) - set([i]):
                        data["train_sen"] = data["train_sen"] + splitted_sentence[j]
                        data["train_class"] = data["train_class"] + splitted_class[j]
                        data["train_concept"] = data["train_concept"] + splitted_concept[j]
                    data["train_sen"], data["train_class"], data["train_concept"] = shuffle(data["train_sen"], data["train_class"], data["train_concept"])
                    params = {
                        "MODALITY": modality,
                        "DATASET": "context_finetuned_MGNC",
                        "SAVE_MODEL": True,
                        "EARLY_STOPPING": True,
                        "EPOCH": 10,
                        "LEARNING_RATE": 0.001,
                        "MAX_SENT_LEN": 237,
                        "MAX_CONCEPT_LEN": 3,
                        "BATCH_SIZE": 5,
                        "WORD_DIM": 200,
                        "CONCEPT_DIM": 10,
                        "VOCAB_SIZE": len(data["vocab"]),
                        "CLASS_SIZE": len(data["classes"]),
                        "CONCEPT_SIZE": len(data["concept"]),
                        "FILTERS": FILTERS,
                        "FILTER_NUM": [100, 100, 100],
                        "FILTER_NUM_CONCEPT": 50,
                        "DROPOUT_PROB": 0.5,
                        "NORM_LIMIT": 3,
                        "GPU": 0,
                        "OPTIMIZATION": "adam",
                        "KNOWLEDGE": "semantic",
                        "DEPTH": False
                    }
                    model, dev_acc, dev_acc_each, listLog = train_ONOFF(data, params, pretrained=True)
                    # print(dev_acc)
                    utilities.save_model(model, params)
                    torch.cuda.empty_cache()
                    average += dev_acc
                strWrite += ("\t" + "{0:.3f}".format(round(average/fold, 3)))
            print(strWrite)
            manipulatedData.write(strWrite + "\n")
    manipulatedData.close()


if __name__ == "__main__":
    # main_shallow()
    fold()
    measure()
