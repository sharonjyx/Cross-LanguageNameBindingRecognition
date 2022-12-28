import os
from nltk import word_tokenize
import pandas as pd
from math import sqrt
import math
import Levenshtein
import re
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
def cosine_similarity(vec1, vec2):
    inner_product = 0
    square_length_vec1 = 0
    square_length_vec2 = 0
    for tup1, tup2 in zip(vec1, vec2):
        inner_product += tup1[1] * tup2[1]
        square_length_vec1 += tup1[1] ** 2
        square_length_vec2 += tup2[1] ** 2
    return (inner_product / sqrt(square_length_vec1 * square_length_vec2))
def edit_Sim(word1, word2):
    """
    按照编辑距离，计算两个词的编辑相似度。单纯使用编辑距离/较长词
    :param word1: 词
    :param word2: 词
    :return: 编辑相似度
    """
    n = max(len(word1), len(word2))
    return 1 - Levenshtein.distance(word1, word2) / n
def leven_Sim(word1, word2):
    """
    计算莱文斯坦比。计算公式r=(sum–ldist)/sum,其中sum是指word1和word2字串的长度总和，ldist是类编辑距离。
    注意这里是类编辑距离，不是通常所说的编辑距离，在类编辑距离中删除、插入依然+1，但是替换+2
    :param word1: 词
    :param word2: 词
    :return: 莱文斯坦比
    """
    return Levenshtein.ratio(word1, word2)
def jaro_Sim(word1, word2):
    """
    计算jaro距离。
    :param word1: 词
    :param word2: 词
    :return: jaro距离
    """
    return Levenshtein.jaro(word1, word2)
def jaroWinkler_Sim(word1, word2):
    """
    计算Jaro–Winkler距离，而Jaro-Winkler则给予了起始部分就相同的字符串更高的分数
    :param word1: 词
    :param word2: 词
    :return: Jaro–Winkler距离
    """
    return Levenshtein.jaro_winkler(word1, word2)
def lcs_Sim(word1, word2):
    """
    value越小，速度越快（value=0.5时，时间=5~10min，慢）
    LCS，计算两个词的最长公共子序列长度。单纯使用lcs/较长词。
    :param word1: 词
    :param word2: 词
    :return: LCS/n
    """
    n = max(len(word1), len(word2))
    dp = [[0 for i in range(len(word2)+1)] for j in range(len(word1)+1)]
    for i in range(len(word1)-1, -1, -1):#倒序，简化边界条件判断
        for j in range(len(word2)-1, -1, -1):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i+1][j+1]+1
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]/n
def dice_Sim(word1, word2):
    """
    计算Dice距离，其用于度量两个集合的相似性，因为可以把字符串理解为一种集合，因此Dice距离也会用于度量字符串的相似性。
    此外，Dice系数的一个非常著名的使用即实验性能评测的F1值
    :param word1: 词
    :param word2: 词
    :return: Dice距离
    """
    a_bigrams = set(word1)
    b_bigrams = set(word2)
    overlap = len(a_bigrams & b_bigrams)
    return overlap*2.0/(len(a_bigrams)+len(b_bigrams))
def similarity_with_2_sents(s1,s2):
    # 分词
    sents = [s1, s2]
    texts = [[word for word in word_tokenize(sent)] for sent in sents]
    # 构建语料库
    all_list = []
    for text in texts:
        all_list += text
    corpus = set(all_list)
    #print(corpus)
    # 对语料库中的单词及标点建立数字映射
    corpus_dict = dict(zip(corpus, range(len(corpus))))
    #print(corpus_dict)
    # 建立句子的向量表示
    def vector_rep(text, corpus_dict):
        vec = []
        for key in corpus_dict.keys():
            if key in text:
                vec.append((corpus_dict[key], text.count(key)))
            else:
                vec.append((corpus_dict[key], 0))
        vec = sorted(vec, key=lambda x: x[0])
        return vec
    vec1 = vector_rep(texts[0], corpus_dict)
    vec2 = vector_rep(texts[1], corpus_dict)
    #print(vec1)
    #print(vec2)
    v1=[]
    v2=[]
    for tup1, tup2 in zip(vec1, vec2):
        v1.append(tup1[1])
        v2.append(tup2[1])
    #print(v1)
    #print(v2)
    v1=np.array(v1)
    v2=np.array(v2)
    # 计算相似度
    #余弦相似度
    cosine_sim.append(cosine_similarity(vec1, vec2))
    #print('两个句子的余弦相似度为： %.4f。' % cosine_similarity(vec1, vec2))
    #欧式距离
    Euclidean_dist.append(np.sqrt(np.sum(np.square(v1 - v2))))
    #print(np.sqrt(np.sum(np.square(v1 - v2))))
    #皮尔逊（pearsonr）相似度
    pearson.append(stats.pearsonr(v1,v2)[0])
    #print(stats.pearsonr(v1,v2)[0])
    #曼哈顿相似度
    manhattanDisSim.append(sum(abs(a - b) for a, b in zip(v1, v2)))
    #print(sum(abs(a - b) for a, b in zip(v1, v2)))
    # #汉明距离
    # Hamming_Dist.append(sum(el1 != el2 for el1, el2 in zip(v1, v2)))
    # #切比雪夫距离
    # Chebyshev_Dist.append(np.abs(v1-v2).max())
def similarity_with_filename(s1,s2):
    a=re.sub('(?=[A-Z])', ' ', s1)
    a=re.sub('(?=[0-9])', ' ', a).strip()
    list1=[i for i in a.split(' ')]
    #print(list1)
    b = re.sub('(?=[A-Z])', ' ', s2)
    b = re.sub('(?=[0-9])', ' ', b).strip()
    list2 = [i for i in b.split(' ')]
    #print(list2)
    # 构建语料库
    all_list = []
    for text in [list1,list2]:
        all_list += text
    corpus = set(all_list)
    #print(corpus)
    # 对语料库中的单词及标点建立数字映射
    corpus_dict = dict(zip(corpus, range(len(corpus))))
    #print(corpus_dict)
    # 建立句子的向量表示
    def vector_rep(text, corpus_dict):
        vec = []
        for key in corpus_dict.keys():
            if key in text:
                vec.append((corpus_dict[key], text.count(key)))
            else:
                vec.append((corpus_dict[key], 0))
        vec = sorted(vec, key=lambda x: x[0])
        return vec
    vec1 = vector_rep(list1, corpus_dict)
    vec2 = vector_rep(list2, corpus_dict)
    v1 = []
    v2 = []
    for tup1, tup2 in zip(vec1, vec2):
        v1.append(tup1[1])
        v2.append(tup2[1])
    #print(v1)
    #print(v2)
    v1 = np.array(v1)
    v2 = np.array(v2)
    # 计算相似度
    # 余弦相似度
    cosine_sim1.append(cosine_similarity(vec1, vec2))
    #print(cosine_similarity(vec1, vec2))
    # 欧式距离
    Euclidean_dist1.append(np.sqrt(np.sum(np.square(v1 - v2))))
    #print(np.sqrt(np.sum(np.square(v1 - v2))))

    s1 = s1.lower()
    s2=re.sub(' ', '', s2).lower()
    # print(s1)
    # print(s2)
    lcsSim.append(lcs_Sim(s1,s2))
    diceSim.append(dice_Sim(s1,s2))
    editSim.append(edit_Sim(s1, s2))
    levenSim.append(leven_Sim(s1, s2))
    jaroSim.append(jaro_Sim(s1, s2))
    jaroWinklerSim.append(jaroWinkler_Sim(s1, s2))
    # print(lcs_Sim(s1,s2))
    # print(dice_Sim(s1,s2))
    # print(edit_Sim(s1, s2))
    # print(leven_Sim(s1, s2))
    # print(jaro_Sim(s1, s2))
    # print(jaroWinkler_Sim(s1, s2))
# s1='de forsthaus backend model IpToCountry long serialVersionUID id ipcIpFrom ipcIpTo String ipcCountryCode2 ipcCountryCode3 ipcCountryName int version'
# s2='de forsthaus backend model IpToCountry id version ipcIpFrom ipcIpTo ipcCountryCode2 ipcCountryCode3 ipcCountryName'
# similarity_with_2_sents(s1, s2)
for i in['itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2','mall','jrecruiter','hispacta','powerstone','jtrac']:
    print(i)
    cosine_sim = []
    Euclidean_dist = []
    pearson = []
    manhattanDisSim = []

    cosine_sim1 = []
    Euclidean_dist1 = []
    lcsSim = []
    diceSim = []
    editSim = []
    levenSim = []
    jaroSim = []
    jaroWinklerSim = []
    os.chdir('E:\\nuaa\\1st_Year\\1_code\\LocalGit\\bert\\'+i)
    df = read_csv("repeattoken.csv",
                  ['field', 'text1', 'text2', 'label', 'javaclass', 'nonjavafile', 'nonjavafilename', 'javapath',
                   'nonjavapath'])
    for index, row in df.iterrows():
        s1 = row['text1']
        s2 = row['text2']
        similarity_with_2_sents(s1, s2)
        s3 = row['javaclass']
        s4 = row['nonjavafile']
        similarity_with_filename(s3, s4)
    df['cosine_sim'] = [v for v in cosine_sim]
    df['Euclidean_dist'] = [v for v in Euclidean_dist]
    pearson = [0 if math.isnan(x) else x for x in pearson]
    df['pearson'] = [v for v in pearson]
    df['manhattanDisSim'] = [v for v in manhattanDisSim]
    # df['Hamming_Dist']=[v for v in Hamming_Dist]
    # df['Chebyshev_Dist']=[v for v in Chebyshev_Dist]
    df['cosine_sim1'] = [v for v in cosine_sim1]
    df['Euclidean_dist1'] = [v for v in Euclidean_dist1]
    df['lcsSim'] = [v for v in lcsSim]
    df['diceSim'] = [v for v in diceSim]
    df['editSim'] = [v for v in editSim]
    df['levenSim'] = [v for v in levenSim]
    df['jaroSim'] = [v for v in jaroSim]
    df['jaroWinklerSim'] = [v for v in jaroWinklerSim]
    df = df.drop(columns="text1")
    df = df.drop(columns="text2")
    df = df.drop(columns="javaclass")
    df = df.drop(columns="nonjavafile")
    df = df.drop(columns="field")
    df = df.drop(columns="nonjavapath")
    df = df.drop(columns="javapath")
    df = df.drop(columns="nonjavafilename")
    df.to_csv('cosine_sim.csv', index=False)