import numpy as np
import torch as t


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    mic_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, mic_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(drug_dis_matrix, drug_matrix, mic_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, mic_matrix))
    return np.vstack((mat1, mat2))



def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = t.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = t.where(t.isinf(D_5), t.full_like(D_5, 0), D_5)
    L_D_11 = t.mm(D_5, L_D_1)
    L_D_11 = t.mm(L_D_11, D_5)
    return L_D_11


def getGipKernel(y, trans, gamma, normalized=False):
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[t.nonzero(k, as_tuple=False)[0]]
    K[t.where(K == 0)] = min_v
    D = t.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S


class Sizes(object):
    def __init__(self, drug_size, mic_size, dis_size):
        self.drug_size = drug_size
        self.mic_size = mic_size
        self.dis_size = dis_size
        self.channel = 64
        self.k_fold = 1
        self.epoch = 10
        self.learn_rate = 0.001
        self.seed = 1
        self.h_gamma = 2 ** (-3)


        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-3)
