from numpy import *


def load_data_set():
    """
    创建实验样本。
    :return: posting_list进行词条切分后的文档集合，class_vec标点符号从文本中去掉。类别标签的集合。
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论。
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现的不重复词的列表。
    :param data_set: 输入的数据。
    :return: 返回一个不重复的词汇表。
    """
    vocab_set = set([])  # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 操作符 | ：用于求两个集合的并集。
    return list(vocab_set)


def set_of_words2_vec(vocab_list, input_set):
    """
    检查词汇表中的单词在输入文档中是否有出现。
    :param vocab_list: 输入的词汇表。
    :param input_set: 输入的某个文档。
    :return: 返回文档向量。向量的每一个元素1或0，分别表示词汇表中的单词在输入文档中是否有出现。
    """
    return_vec = [0] * len(vocab_list)  # 创建一个和词汇表等长的向量，并将其元素设置为0。
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设置为1。
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
            # index() 函数，如果存在返回索引值，如果不存在，报错。
            # 注意与find() 函数的区别。
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return return_vec


# def bag_of_words2_vec_MN(vocab_list, input_set):
#     """
#     检查词汇表中的单词在输入文档中是否有出现。
#     :param vocab_list: 输入的词汇表。
#     :param input_set: 输入的某个文档。
#     :return: 返回文档向量。向量的每一个元素1或0，分别表示词汇表中的单词在输入文档中是否有出现。
#     """
#     return_vec = [0] * len(vocab_list)  # 创建一个和词汇表等长的向量，并将其元素设置为0。
#     # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设置为1。
#     for word in input_set:
#         if word in vocab_list:
#             return_vec[vocab_list.index(word)] += 1
#             # index() 函数，如果存在返回索引值，如果不存在，报错。
#             # 注意与find() 函数的区别。
#     return return_vec


def train_NB0(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练函数
    :param train_matrix: 文档矩阵，这个文档矩阵指的是 set_of_words2_vec() 函数中得到向量的并集。
    :param train_category: 由每篇文档所构成的向量
    :return:
    """
    num_train_docs = len(train_matrix)  # 向量的个数，也就是训练文档的篇数
    num_words = len(train_matrix[0])  # 词汇表的所有单词的总个数
    p_abusive = sum(train_category) / float(num_train_docs)  # (0+1+0+1+0+1) / 6 = 0.5
    # 初始化概率，初始化程序中的分子变量和分母变量
    # p0_num = zeros(num_words)
    # p1_num = zeros(num_words)
    # 防止因为一个概率值为0，最后求得的乘积也为0
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    # 分母变量是一个元素个数等于词汇表大小的NumPy数组。
    # p0_denom = 0.0
    # p1_denom = 0.0
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]  # 被分类为1的文档中，各个词汇出现的总次数
            p1_denom += sum(train_matrix[i])  # 被分类为1的文档中的单词的总个数。
        else:
            p0_num += train_matrix[i]  # 被分类为0的文档中，各个词汇出现的总次数
            p0_denom += sum(train_matrix[i])  # 被分类为0的文档中的单词的总个数。
    # p1_vect = p1_num / p1_denom
    # p0_vect = p0_num / p0_denom
    # 求得的概率太小，导致四舍五入后变为0。采用对数可以避免下溢出或者浮点数舍入导致的错误。
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classifyNB(vec2_classify, p0_vec, p1_vec, p_class1):
    """

    :param vec2_classify: 要分类的向量，通过set_of_words2_vec() 所求得的向量
    :param p0_vec:
    :param p1_vec:
    :param p_class1:
    上述三项均为train_NB0() 所求得的三个概率
    :return:
    """
    # 对应元素相乘，
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_NB():
    """
    便利函数：封装所有操作，以节省代码时间。
    就是一个测试函数。直接输出就行。
    :return:
    """
    list_o_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_o_posts)
    train_mat = []
    for post_in_doc in list_o_posts:
        train_mat.append(set_of_words2_vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_NB0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as:', classifyNB(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as:', classifyNB(this_doc, p0_v, p1_v, p_ab))


def text_parse(big_string):
    """
    接收一个大字符串并将其解析为字符串列表
    :param big_string: 大字符串
    :return: 字符串列表
    """
    import re  # 正则表达式
    list_of_tokens = re.split('\W', big_string)  # \W 用来去掉空字符串
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    """
    对贝叶斯垃圾邮件分类器进行自动化处理
    :return:
    """
    # 导入并解析文本文档
    doc_list = []  # 所有文档
    class_list = []  # 所有文档对应的类别
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)  # list 中的 extend() 用于追加列表
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    # 随机构建训练集
    vocab_list = create_vocab_list(doc_list)
    training_set = list(range(50))  # python3 中，range返回的是range对象，不返回数组对象
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内。
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(set_of_words2_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_NB0(array(train_mat), array(train_classes))
    # 对测试集进行分类
    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words2_vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is:', float(error_count) / len(test_set))
