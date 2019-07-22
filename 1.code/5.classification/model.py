import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN, self).__init__()
#
#         self.MODALITY = kwargs["MODALITY"]
#         self.BATCH_SIZE = kwargs["BATCH_SIZE"]
#         self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
#         self.WORD_DIM = kwargs["WORD_DIM"]
#         self.CONCEPT_SIZE = kwargs["CONCEPT_SIZE"]
#         self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
#         self.CLASS_SIZE = kwargs["CLASS_SIZE"]
#         self.FILTER_PRE = kwargs["FILTER_PRE"]
#         self.FILTER_NUM_PRE = kwargs["FILTER_NUM_PRE"]
#         self.FILTERS = kwargs["FILTERS"]
#         self.FILTER_NUM = kwargs["FILTER_NUM"]
#         self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
#         self.IN_CHANNEL = 1
#
#         assert (len(self.FILTERS) == len(self.FILTER_NUM))
#
#         if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
#             self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#             self.WV_MATRIX = kwargs["WV_MATRIX"]
#             self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
#         if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
#             self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#             self.WVF_MATRIX = kwargs["WVF_MATRIX"]
#             self.embedding2.weight.data.copy_(torch.from_numpy(self.WVF_MATRIX))
#         if self.MODALITY in set(["NK", "DK", "NDK"]):
#             self.embedding3 = nn.Embedding(self.CONCEPT_SIZE + 1, self.WORD_DIM, padding_idx=self.CONCEPT_SIZE)
#             self.CONCEPT_MATRIX = kwargs["CONCEPT_MATRIX"]
#             self.embedding3.weight.data.copy_(torch.from_numpy(self.CONCEPT_MATRIX))
#
#         if self.MODALITY in set(["ND", "NK", "DK"]):
#             self.IN_CHANNEL = 2
#         elif self.MODALITY in set(["NDK"]):
#             self.IN_CHANNEL = 3
#
#         conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM_PRE, self.WORD_DIM * self.FILTER_PRE, stride=self.WORD_DIM)
#         nn.init.xavier_normal_(conv.weight)
#         setattr(self, f'conv_pre', conv)
#
#         for i in range(len(self.FILTERS)):
#             conv = nn.Conv1d(self.FILTER_NUM_PRE, self.FILTER_NUM[i], self.FILTERS[i], stride=1)
#             nn.init.xavier_normal_(conv.weight)
#             setattr(self, f'conv_{i}', conv)
#
#         # for i in range(len(self.FILTERS1)):
#         #     self.layer1 = nn.Sequential(
#         #         nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM1[i], self.WORD_DIM * self.FILTERS1[i], stride=self.WORD_DIM),
#         #         nn.BatchNorm1d(self.FILTER_NUM1[i]),
#         #         nn.ReLU(),
#         #         nn.MaxPool1d()
#         #     )
#         #
#         # for i in range(len(self.FILTERS2)):
#         #     self.layer2 = nn.Sequential(
#         #         nn.Conv1d(self.FILTER_NUM1[i], self.FILTER_NUM2[i], self.FILTERS2[i], stride=1),
#         #         nn.BatchNorm1d(self.FILTER_NUM2[i]),
#         #         nn.ReLU(),
#         #         nn.MaxPool1d()
#         #     )
#
#         self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
#         nn.init.xavier_normal_(self.fc.weight)
#
#     def get_conv_pre(self):
#         return getattr(self, f'conv_pre')
#
#     def get_conv(self, i):
#         return getattr(self, f'conv_{i}')
#
#     def forward(self, inp):
#         if self.MODALITY in set(["N", "D", "ND", "NK", "DK", "NDK"]):
#             boolNgram = False
#             if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
#                 x = self.embedding(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#                 boolNgram = True
#             if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
#                 if boolNgram:
#                     x2 = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#                     x = torch.cat((x, x2), 1)
#                 else:
#                     x = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#             if self.MODALITY in set(["NK", "DK", "NDK"]):
#                 x3 = self.embedding3(inp[1]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#                 x = torch.cat((x, x3), 1)
#
#             conv_result_pre = F.relu(self.get_conv_pre()(x))
#             conv_result = [
#                 F.max_pool1d(F.relu(self.get_conv(i)(conv_result_pre)),
#                              self.MAX_SENT_LEN - self.FILTERS[i] - self.FILTER_PRE + 2).view(-1, self.FILTER_NUM[i]) for i
#                 in range(len(self.FILTERS))]
#
#             x = torch.cat(conv_result, 1)
#             x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
#             x = self.fc(x)
#             return x
#         else:
#             print(self.MODALITY)


# class CNN_shallow(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN_shallow, self).__init__()
#
#         self.MODALITY = kwargs["MODALITY"]
#         self.BATCH_SIZE = kwargs["BATCH_SIZE"]
#         self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
#         self.WORD_DIM = kwargs["WORD_DIM"]
#         self.CONCEPT_SIZE = kwargs["CONCEPT_SIZE"]
#         self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
#         self.CLASS_SIZE = kwargs["CLASS_SIZE"]
#         self.FILTERS = kwargs["FILTERS"]
#         self.FILTER_NUM = kwargs["FILTER_NUM"]
#         self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
#         self.IN_CHANNEL = 1
#
#         assert (len(self.FILTERS) == len(self.FILTER_NUM))
#
#         # one for UNK and one for zero padding
#         # if self.MODEL == "static":
#         #     self.embedding.weight.requires_grad = False
#
#
#         if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
#             self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#             self.WV_MATRIX = kwargs["WV_MATRIX"]
#             self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
#         if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
#             self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#             self.WVF_MATRIX = kwargs["WVF_MATRIX"]
#             self.embedding2.weight.data.copy_(torch.from_numpy(self.WVF_MATRIX))
#         if self.MODALITY in set(["NK", "DK", "NDK"]):
#             self.embedding3 = nn.Embedding(self.CONCEPT_SIZE + 1, self.WORD_DIM, padding_idx=self.CONCEPT_SIZE)
#             self.CONCEPT_MATRIX = kwargs["CONCEPT_MATRIX"]
#             self.embedding3.weight.data.copy_(torch.from_numpy(self.CONCEPT_MATRIX))
#
#         if self.MODALITY in set(["ND", "NK", "DK"]):
#             self.IN_CHANNEL = 2
#         elif self.MODALITY in set(["NDK"]):
#             self.IN_CHANNEL = 3
#
#         for i in range(len(self.FILTERS)):
#             conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
#             setattr(self, f'conv_{i}', conv)
#
#         self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
#
#     def get_conv(self, i):
#         return getattr(self, f'conv_{i}')
#
#     def forward(self, inp):
#         boolNgram = False
#         if self.MODALITY in set(["N", "ND", "NK", "NDK"]):
#             x = self.embedding(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#             boolNgram = True
#         if self.MODALITY in set(["D", "ND", "DK", "NDK"]):
#             if boolNgram:
#                 x2 = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#                 x = torch.cat((x, x2), 1)
#             else:
#                 x = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#         if self.MODALITY in set(["NK", "DK", "NDK"]):
#             x3 = self.embedding3(inp[1]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#             x = torch.cat((x, x3), 1)
#         conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
#
#         x = torch.cat(conv_results, 1)
#         x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
#         x = self.fc(x)
#         return x


# class CNN_PI(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN_PI, self).__init__()
#
#         self.MODEL = kwargs["MODEL"]
#         self.BATCH_SIZE = kwargs["BATCH_SIZE"]
#         self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
#         self.WORD_DIM = kwargs["WORD_DIM"]
#         self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
#         self.CLASS_SIZE = kwargs["CLASS_SIZE"]
#         self.FILTER_PRE = kwargs["FILTER_PRE"]
#         self.FILTER_NUM_PRE = kwargs["FILTER_NUM_PRE"]
#         self.FILTERS = kwargs["FILTERS"]
#         self.FILTER_NUM = kwargs["FILTER_NUM"]
#         self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
#         self.IN_CHANNEL = 1
#
#         assert (len(self.FILTERS) == len(self.FILTER_NUM))
#
#         # one for UNK and one for zero padding
#         self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#         if self.MODEL in set(["static", "non-static", "multichannel"]):
#             self.WV_MATRIX = kwargs["WV_MATRIX"]
#             self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
#             if self.MODEL == "static":
#                 self.embedding.weight.requires_grad = False
#             elif self.MODEL == "multichannel":
#                 self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#                 self.WVF_MATRIX = kwargs["WVF_MATRIX"]
#                 self.embedding2.weight.data.copy_(torch.from_numpy(self.WVF_MATRIX))
#                 self.IN_CHANNEL = 2
#
#         conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM_PRE, self.WORD_DIM * self.FILTER_PRE, stride=self.WORD_DIM)
#         nn.init.xavier_normal_(conv.weight)
#         setattr(self, f'conv_pre', conv)
#
#         for i in range(len(self.FILTERS)):
#             conv = nn.Conv1d(self.FILTER_NUM_PRE, self.FILTER_NUM[i], self.FILTERS[i], stride=1)
#             nn.init.xavier_normal_(conv.weight)
#             setattr(self, f'conv_{i}', conv)
#
#         self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
#         nn.init.xavier_normal_(self.fc.weight)
#
#     def get_conv_pre(self):
#         return getattr(self, f'conv_pre')
#
#     def get_conv(self, i):
#         return getattr(self, f'conv_{i}')
#
#     def forward(self, inp):
#         x = self.embedding(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#         # print("input ling size: ", inp[0].size())
#         # print("input bio size: ", inp[1].size())
#
#         if self.MODEL == "multichannel":
#             x2 = self.embedding2(inp[0]).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#             x = torch.cat((x, x2), 1)
#             # print("x/x2: ", x.size())
#
#         conv_result_pre = F.relu(self.get_conv_pre()(x))
#
#         conv_result = [
#             F.max_pool1d(F.relu(self.get_conv(i)(conv_result_pre)),
#                          self.MAX_SENT_LEN - self.FILTERS[i] - self.FILTER_PRE + 2).view(-1, self.FILTER_NUM[i]) for i
#             in range(len(self.FILTERS))]
#
#         # conv_results = [
#         #     F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i]
#         #                  + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
#
#         x = torch.cat(conv_result, 1)
#         x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
#         x = self.fc(x)
#
#         return x
#
#
# class CNN_PI_shallow(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN_PI_shallow, self).__init__()
#
#         self.MODEL = kwargs["MODEL"]
#         self.BATCH_SIZE = kwargs["BATCH_SIZE"]
#         self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
#         self.WORD_DIM = kwargs["WORD_DIM"]
#         self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
#         self.CLASS_SIZE = kwargs["CLASS_SIZE"]
#         self.FILTERS = kwargs["FILTERS"]
#         self.FILTER_NUM = kwargs["FILTER_NUM"]
#         self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
#         self.IN_CHANNEL = 1
#
#         assert (len(self.FILTERS) == len(self.FILTER_NUM))
#
#         # one for UNK and one for zero padding
#         self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#         if self.MODEL == "multichannel" or self.MODEL == "non-static" or self.MODEL == "static":
#             # self.WV_MATRIX = kwargs["WV_MATRIX"]
#             self.WV_MATRIX = kwargs["WVF_MATRIX"]
#             self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
#             if self.MODEL == "static":
#                 self.embedding.weight.requires_grad = False
#             elif self.MODEL == "multichannel":
#                 self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
#                 self.WVF_MATRIX = kwargs["WVF_MATRIX"]
#                 self.embedding2.weight.data.copy_(torch.from_numpy(self.WVF_MATRIX))
#                 self.IN_CHANNEL = 2
#
#         for i in range(len(self.FILTERS)):
#             conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
#             setattr(self, f'conv_{i}', conv)
#
#         self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
#
#     def get_conv(self, i):
#         return getattr(self, f'conv_{i}')
#
#     def forward(self, inp):
#         x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#         if self.MODEL == "multichannel":
#             x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
#             x = torch.cat((x, x2), 1)
#
#         conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
#
#         x = torch.cat(conv_results, 1)
#         x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
#         x = self.fc(x)
#
#         return x


class MGNC_CNN_shallow(nn.Module):
    def __init__(self, **kwargs):
        super(MGNC_CNN_shallow, self).__init__()

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


class depthwiseCNN(nn.Module):
    def __init__(self, **kwargs):
        super(depthwiseCNN, self).__init__()

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
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM_CONCEPT, 3, stride=3)
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
            xk = torch.transpose(self.embedding3(inp[1]), 1, 2).contiguous().view(-1, 1, self.CONCEPT_DIM * 3)
            conv_result_k = [F.max_pool1d(F.relu(self.get_fn("bn_k_0")(self.get_fn("conv_k_0")(xk))), self.CONCEPT_DIM).view(-1, self.FILTER_NUM_CONCEPT)]
        x = torch.cat(conv_result_n + conv_result_d + conv_result_k, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        return x