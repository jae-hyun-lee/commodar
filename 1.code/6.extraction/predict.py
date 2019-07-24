import time
import utilities_predict
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from model_predict import MGNC_CNN
init_time = time.time()


def global_measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


def temp_measure(temp_time):
    after_time = time.time()
    dif_time = after_time - temp_time
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


def test(data, model, params):
    model.eval()
    test_sen, test_concept, test_id = data["sen"], data["triplet"], data["id"]
    test_sen = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent]
                + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)) for sent in test_sen]
    test_sen = Variable(torch.LongTensor(test_sen)).cuda(params["GPU"])
    test_concept = [[data["concept_to_idx"][concept] if concept in data["concept"] else params["CONCEPT_SIZE"]
                     for concept in seq] + [params["CONCEPT_SIZE"]] * (params["MAX_CONCEPT_LEN"] - len(seq))
                    for seq in test_concept]
    test_concept = Variable(torch.LongTensor(test_concept)).cuda(params["GPU"])
    try:
        pred = np.argmax(model([test_sen, test_concept]).cpu().data.numpy(), axis=1)
        dic_class_label = {0: 'Binding|NA', 1: 'Decrease|Backward', 2: 'Decrease|Forward', 3: 'Increase|Backward', 4: 'Increase|Forward', 5: 'False', 6: 'Regulation|Backward', 7: 'Regulation|Forward'}
        return [data["id"][i] + "\t" + dic_class_label[pred[i]] for i in range(len(data["id"]))]
    except RuntimeError:
        print("CUDA out of memory")
        return ["CUDA out of memory"]


def oov(data, fine_tuned_state, params, new_vocab, new_concept):
    from gensim.models.keyedvectors import KeyedVectors

    word_to_idx = utilities_predict.load_dictionary("word_to_idx_fine-tuning")
    idx_to_word = utilities_predict.load_dictionary("idx_to_word_fine-tuning")
    concept_to_idx = utilities_predict.load_dictionary("concept_to_idx_fine-tuning")
    idx_to_concept = utilities_predict.load_dictionary("idx_to_concept_fine-tuning")

    loaded_vocab = set(word_to_idx.keys())
    idx = len(loaded_vocab)
    word_vectors = KeyedVectors.load_word2vec_format("n-gram.model.bin", binary=True)
    wv_matrix = []
    word_dep_vocab, word_dep_vectors = utilities_predict.npy_load("dependency")
    wvf_matrix = []

    for vocab in new_vocab:
        if vocab not in loaded_vocab:
            boolN = True if vocab in word_vectors.vocab else False
            boolD = True if vocab in word_dep_vocab else False

            if boolN or boolD:
                word_to_idx[vocab] = idx
                idx_to_word[idx] = vocab
                idx += 1
                if boolN:
                    wv_matrix.append(word_vectors.word_vec(vocab))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, 200).astype("float16"))

                if boolD:
                    wvf_matrix.append(word_dep_vectors[word_dep_vocab[vocab]])
                else:
                    wvf_matrix.append(np.random.uniform(-0.01, 0.01, 200).astype("float16"))

    loaded_concept = set(concept_to_idx.keys())
    idx = len(loaded_concept)
    semantic_vocab, semantic_vectors = utilities_predict.npy_load("knowledge_triplet")
    concept_matrix = []

    for con in new_concept:
        if con not in loaded_concept:
            concept_to_idx[con] = idx
            idx_to_concept[idx] = con
            idx += 1
            vector = np.mean([semantic_vectors[semantic_vocab[sType]] for sType in con.split("|")], axis=0)
            concept_matrix.append(vector)

    data["concept_to_idx"] = concept_to_idx
    data["idx_to_concept"] = idx_to_concept
    data["word_to_idx"] = word_to_idx
    data["idx_to_word"] = idx_to_word

    params["VOCAB_SIZE"] = 55936 + len(wv_matrix)
    params["CONCEPT_SIZE"] = 115 + len(concept_matrix)

    new_model = MGNC_CNN(**params)
    new_model_state = new_model.state_dict()
    # print(len(wv_matrix), len(concept_matrix))

    if (len(wv_matrix) == 0) & (len(concept_matrix) == 0):
        # print("no novel vocabs and concepts")
        new_model.load_state_dict(fine_tuned_state)
        return data, new_model

    if len(wv_matrix) != 0:
        npEmbedding1 = wv_matrix
        npEmbedding1 = np.concatenate([npEmbedding1, [np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16")]], axis=0)
        npEmbedding1 = np.concatenate([npEmbedding1, [np.zeros(params["WORD_DIM"]).astype("float16")]], axis=0)
        new_model_state['embedding1.weight'] = torch.cat((fine_tuned_state['embedding1.weight'][:-2].to("cpu"), torch.from_numpy(npEmbedding1)), 0)

        npEmbedding2 = wvf_matrix
        npEmbedding2 = np.concatenate([npEmbedding2, [np.random.uniform(-0.01, 0.01, params["WORD_DIM"]).astype("float16")]], axis=0)
        npEmbedding2 = np.concatenate([npEmbedding2, [np.zeros(params["WORD_DIM"]).astype("float16")]], axis=0)
        new_model_state['embedding2.weight'] = torch.cat((fine_tuned_state['embedding2.weight'][:-2].to("cpu"), torch.from_numpy(npEmbedding2).float()), 0)

    if len(concept_matrix) != 0:
        npEmbedding3 = concept_matrix
        npEmbedding3 = np.concatenate([npEmbedding3, [np.zeros(params["CONCEPT_DIM"]).astype("float16")]], axis=0)
        new_model_state['embedding3.weight'] = torch.cat((fine_tuned_state['embedding3.weight'][:-1].to("cpu"), torch.from_numpy(npEmbedding3)), 0)

    new_model.load_state_dict(new_model_state)
    return data, new_model


def predict(file, fine_tuned_state, params):
    data = utilities_predict.read_context_unlabeled(file)
    if len(data["sen"]) == 0:
        writeOutput(["empty"], file.replace("resource", "result"))
    else:
        params["MAX_SENT_LEN"] = max([len(sen) for sen in data["sen"]])
        new_vocab = sorted(list(set([w for sent in data["sen"] for w in sent])))
        new_concept = sorted(list(set([c for concept in data["triplet"] for c in concept])))
        data, model = oov(data, fine_tuned_state, params, new_vocab, new_concept)

        data["vocab"] = sorted(data["word_to_idx"].keys())
        data["concept"] = sorted(data["concept_to_idx"].keys())

        model = model.to('cuda')
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        result = test(data, model, params)
        if "CUDA out of memory" in result[0]:
            memory_error_file_split(file, 4)
        writeOutput(result, file.replace("resource", "result"))


def memory_error_file_split(strResource, split):
    with open(strResource, "r", encoding="utf-8") as fileResource:
        listContents = [stringCleansing(line) for line in fileResource.readlines()]
        for i in range(split):
            writeOutput(listContents[int(len(listContents) * i / split):int(len(listContents) * (i+1) / split)],
                        strResource.replace(".tsv", "_" + str(i+1) + "of" + str(split) + ".tsv"))


def predict_batch():
    params = {
        "MODALITY": "NDK",
        "MAX_CONCEPT_LEN": 3,
        "WORD_DIM": 200,
        "CONCEPT_DIM": 10,
        "CLASS_SIZE": 8,
        "FILTERS": [21, 22, 23],
        "FILTER_NUM": [100, 100, 100],
        "FILTER_NUM_CONCEPT": 50,
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": 0
    }
    filter_size = ",".join([str(size) for size in params['FILTERS']])
    fine_tuned_path = f"classification_fine-tuning_NDK_{filter_size}.pt"
    fine_tuned_state = torch.load(fine_tuned_path)

    infile = "[INPUT SENTENCE TSV HERE]"
    temp_time = time.time()
    predict(infile, fine_tuned_state, params)
    torch.cuda.empty_cache()
    temp_measure(temp_time)


if __name__ == '__main__':
    predict_batch()
    global_measure()
