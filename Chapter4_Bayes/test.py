# from bayes import *
#
# list_o_posts, list_classes = load_data_set()
# my_vocab_list = create_vocab_list(list_o_posts)
# print(my_vocab_list)
#
# print(set_of_words2_vec(my_vocab_list, list_o_posts[0]))
# from numpy import *
# import bayes
# list_o_posts, list_classes = bayes.load_data_set()
# my_vocab_list = bayes.create_vocab_list(list_o_posts)
# train_mat = []
# for post_in_doc in list_o_posts:
#      train_mat.append(bayes.set_of_words2_vec(my_vocab_list, post_in_doc))
#
# p0_v, p1_v, p_ab = bayes.train_NB0(train_mat, list_classes)
# print(train_mat)

# import bayes

# bayes.testing_NB()
# my_sent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
# import re
#
# reg_ex = re.compile('\W')
# list_of_tokens = reg_ex.split(my_sent)
# print(list_of_tokens)
# email_text = open('email/ham/6.txt').read()
# import bayes

# print(bayes.text_parse(email_text))


import bayes
bayes.spam_test()
