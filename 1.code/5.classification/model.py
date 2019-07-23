import torch
import torch.nn as nn
import torch.nn.functional as F

class MGNC_CNN(nn.Module):
    def __init__(self, **kwargs):
        super(MGNC_CNN, self).__init__()

        self.MODALITY = kwargs["MODALITY"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.CONCEPT_DIM = kwargs["CONCEPT_DIM"]
        self.CONCEPT_SIZE = kwargs["CONCEPT_SIZE"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.FILTER_NUM_CONCEPT = kwargs["FILTER_NUM_CONCEPT"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
            self.embedding1 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding1.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
            self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
            self.WVF_MATRIX = kwargs["WVF_MATRIX"]
            self.embedding2.weight.data.copy_(torch.from_numpy(self.WVF_MATRIX))
        if self.MODALITY in set(["NK", "DK", "NDK"]):
            self.embedding3 = nn.Embedding(self.CONCEPT_SIZE + 1, self.CONCEPT_DIM, padding_idx=self.CONCEPT_SIZE)
            self.CONCEPT_MATRIX = kwargs["CONCEPT_MATRIX"]
            self.embedding3.weight.data.copy_(torch.from_numpy(self.CONCEPT_MATRIX))
            # self.embedding3.weight.requires_grad = False

        num_infeature_fc = 0
        if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
            for i in range(len(self.FILTERS)):
                conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
                nn.init.xavier_normal_(conv.weight)
                setattr(self, f'conv_n_{i}', conv)
                bn = nn.BatchNorm1d(self.FILTER_NUM[i])
                setattr(self, f'bn_n_{i}', bn)
                num_infeature_fc += self.FILTER_NUM[i]

        if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
            for i in range(len(self.FILTERS)):
                conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
                nn.init.xavier_normal_(conv.weight)
                setattr(self, f'conv_d_{i}', conv)
                bn = nn.BatchNorm1d(self.FILTER_NUM[i])
                setattr(self, f'bn_d_{i}', bn)
                num_infeature_fc += self.FILTER_NUM[i]

        if self.MODALITY in set(["NK", "DK", "NDK"]):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM_CONCEPT, self.CONCEPT_DIM * 3, stride=self.CONCEPT_DIM)
            nn.init.xavier_normal_(conv.weight)
            setattr(self, 'conv_k_0', conv)
            bn = nn.BatchNorm1d(self.FILTER_NUM_CONCEPT)
            setattr(self, f'bn_k_0', bn)
            num_infeature_fc += self.FILTER_NUM_CONCEPT

        self.fc = nn.Linear(num_infeature_fc, self.CLASS_SIZE)
        setattr(self, 'fc', self.fc)

    def get_fn(self, fn):
        return getattr(self, fn)

    def forward(self, inp):
        conv_result_n = []
        conv_result_d = []
        conv_result_k = []

        if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
            xn = self.embedding1(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            conv_result_n = [F.max_pool1d(F.relu(self.get_fn(f"bn_n_{i}")(self.get_fn(f"conv_n_{i}")(xn))), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
            xd = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            conv_result_d = [F.max_pool1d(F.relu(self.get_fn(f"bn_d_{i}")(self.get_fn(f"conv_d_{i}")(xd))), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        if self.MODALITY in set(["NK", "DK", "NDK"]):
            xk = self.embedding3(inp[1]).view(-1, 1, self.CONCEPT_DIM * 3)
            conv_result_k = [F.relu(self.get_fn("bn_k_0")(self.get_fn("conv_k_0")(xk))).view(-1, self.FILTER_NUM_CONCEPT)]

        x = torch.cat(conv_result_n + conv_result_d + conv_result_k, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        return x
