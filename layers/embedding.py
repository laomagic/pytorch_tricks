

import numpy as np
import jieba_fast as jieba
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """相对的位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, x.size(1)], requires_grad=False)
        return self.dropout(x)



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionEmbedding(max_position_embeddings, embedding_dim):
    m = nn.Embedding(max_position_embeddings, embedding_dim)
    nn.init.normal_(m.weight, 0, 0.1)
    return m


class MixEmbedding(nn.Module):
    """
    字向量和词向量的混合
    @:param path 词向量的路径
    @:param emb_dim 混合向量的维度
    @:param vocab_size 字向量的词表大小
    @:return 混合向量
    """

    def __init__(self, path, emb_dim, vocab_size):
        super(MixEmbedding, self).__init__()
        self.word_embedding, self.word2id, self.id2word, self.word_size = self.load_word_vector(path)
        self.word_dense = nn.Linear(self.word_size, emb_dim, bias=False)
        self.char_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.word_dense.weight)
        nn.init.normal_(self.char_embedding.weight, 0, 0.1)

    @staticmethod
    def load_word_vector(path,requires_grad=False):
        word2vec = Word2Vec.load(path)
        id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
        word2id = {j: i for i, j in id2word.items()}
        word2vec = word2vec.wv.vectors  # 所有的向量值
        word_size = word2vec.shape[1]  # 向量的维度
        vocab_size = word2vec.shape[0]  # 词表的大小
        word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])
        emb = nn.Embedding(vocab_size, word_size).from_pretrained(torch.tensor(word2vec, dtype=torch.float))  # 词向量
        emb.weight.requires_grad = requires_grad  # 梯度不更新
        return emb, word2id, id2word, word_size

    def forward(self, input_ids):
        char_ids, word_ids = input_ids
        assert len(char_ids) == len(word_ids)  # ids长度相等
        word_emb = self.word_embedding(word_ids)
        word_emb = self.word_dense(word_emb)  # 向量维度转换
        char_emb = self.char_embedding(char_ids)
        return char_emb + word_emb
    
    
if __name__ == '__main__':
    path = '../word2vec_baike/word2vec_baike'
    mix_emb = MixEmbedding(path, 768, 50000)
    text = "生命至上，安全第一。"
    words = jieba.lcut(text)
    # words_ids = [mix_emb.word2id.get(word, 0) for word in words]
    sents.append(words)
    word_ids = [mix_emb.word2id.get(word, 0) for word in words for _ in word]  # word_id 对齐->char_id
    # words_ids = sent2vec(sents, mix_emb.word2id)  # id 对齐
    char_ids = [mix_emb.word2id.get(char, 0) for char in text]
    input_ids = torch.tensor([char_ids, word_ids])
    s = mix_emb(input_ids)
    # emb_dim = 64
    # max_seq_len = 100
    # seq_len = 20
    # DEVICE = torch.device("cpu")
    # pe = PositionalEncoding(emb_dim, 0, max_seq_len)
    # x = torch.zeros(1, seq_len, emb_dim, device=DEVICE)
    # positional_encoding = pe(x)
    # plt.figure()
    # sns.heatmap(positional_encoding.squeeze().to("cpu"))
    # plt.xlabel("i")
    # plt.ylabel("pos")
    # plt.show()
    #
    # plt.figure()
    # y = positional_encoding.to("cpu").numpy()
    # plt.plot(np.arange(seq_len), y[0, :, 0: 64: 8], ".")
    # plt.legend(["dim %d" % p for p in [0, 7, 15, 31, 63]])
    # plt.show()