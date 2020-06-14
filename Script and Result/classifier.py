# ---------------------------------------------------------------------
# Project 2
# Written by Vida Abdollahi 40039052
# For COMP 6721 Section FK – Fall 2019
# ---------------------------------------------------------------------

import math
from typing import List, Any, Union

import nltk
import pandas as pd
import string
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize, word_tokenize, FreqDist
import re
from nltk.stem import WordNetLemmatizer

#------------------Class Word----------------------------#
class WordClass:

    def __init__(self, word, frequency, probability):
        self.frequency = frequency
        self.word = word
        self.probability = probability

#------------------Reading CSV files----------------------------#

# region Description
train_dataset = pd.read_csv('hn2018_2019.csv', nrows=276982)
column_contain = ['Object ID','Title', 'Post Type']
type = train_dataset['Post Type']
df = pd.DataFrame(data = train_dataset, columns=column_contain)


test_dataset = pd.read_csv('hn2018_2019.csv', skiprows=[i for i in range(1,376982)])
column_contain2 = ['Object ID','Title', 'Post Type']
df_test = pd.DataFrame(data = test_dataset, columns=column_contain2)

df_test['Compare'] = 'Default' # wrong g or right
df_test['Result'] = 'Unknown' # result of the classifier


alpha = 0.5


story_type = df.loc[df['Post Type'] == 'story']
story_count = story_type['Post Type'].value_counts()

ask_hn_type = df.loc[df['Post Type'] == 'ask_hn']
ask_count = ask_hn_type['Post Type'].value_counts()

poll_type = df.loc[df['Post Type'] == 'poll']
poll_count = poll_type['Post Type'].value_counts()

show_hn_type = df.loc[df['Post Type'] == 'show_hn']
show_count = show_hn_type['Post Type'].value_counts()

sum_of_types = df.shape[0]

probability_story = story_count / sum_of_types
probability_ask = ask_count / sum_of_types
probability_poll = poll_count /sum_of_types
probability_show = show_count / sum_of_types


def stop_words(data):

    f = open("Stopwords", "r")
    words = f.read()
    stop_words = list(words)

    filtered_sentence = []
    for w in data:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence


def preprocessing(_type):

    a = _type['Title'].str.lower().str.cat(sep=' ')
    b = re.sub('[^A-Za-z_]+', ' ', a)
    word_tokens = word_tokenize(b)
    #word_tokens = word_tokenize()
    lemmatizer = WordNetLemmatizer()
    cleaned_data_title = [word for word in word_tokens if not word.isnumeric()]
    data = [lemmatizer.lemmatize(word) for word in cleaned_data_title]
    #data_length_filter = [word for word in data if len(word) < 9 and len(word) > 2]
    #data_stop_words_filter = stop_words(data)
    return data



def calculation(cleaned_data_title):

    cleaned_main_data_all = preprocessing(df)

    # Number of all words in only "story" type - but this is repetative, like 'a' must be repeated 10000 times!
    all_words_count  = len(cleaned_data_title)
    main_words_count = len(cleaned_main_data_all)

    word_dist = nltk.FreqDist(cleaned_data_title)
    result = pd.DataFrame(word_dist.most_common(),
                        columns = ['Word', 'Frequency'])

    # this dictionary is like ["word": word_object]
    dictionary = {}
    # Number of unique words in only "story" type
    row_count = result.shape[0]

    #beta = row_count * row_count


    word_dist_all = nltk.FreqDist(cleaned_main_data_all)
    result_all = pd.DataFrame(word_dist_all.most_common(),
                          columns=['Word', 'Frequency'])
    # Number of unique words in the whole database
    beta = result_all.shape[0]

    # write all the tokenized words to the file
    sorted_result = result_all.sort_values('Word')


    f = open("vocabulary.txt", "w+")
    for index in sorted_result.index:
        f.write('{}  {}' .format(result_all['Word'][index], result_all['Frequency'][index] ))
        f.write("\n")
    f.close()


    # initializing dictionary of all words in the main dataset with default probability value
    for word in cleaned_main_data_all:
        dictionary[word] = WordClass(word, 0, alpha/(all_words_count + beta * alpha))

    for word in cleaned_data_title:
        # frequency of each words is when we only have "story" type
        frequency = word_dist[word]
        # probability (word|story) is frequency of that word in story to number of unique words in story
        probability = (frequency + alpha)/ (all_words_count + beta * alpha)
        _word = WordClass(word, word_dist[word], probability)
        # add object to dictionary
        dictionary[word] = _word
    return dictionary



cleaned_data_title_story = preprocessing(story_type)
dictionary_story = calculation(cleaned_data_title_story)

cleaned_data_title_show_hn = preprocessing(show_hn_type)
dictionary_show_hn = calculation(cleaned_data_title_show_hn)

cleaned_data_title_poll = preprocessing(poll_type)
dictionary_poll = calculation(cleaned_data_title_poll)

cleaned_data_title_ask_hn = preprocessing(ask_hn_type)
dictionary_ask_hn = calculation(cleaned_data_title_ask_hn)

f= open("model.txt","w+")

cleaned_main_data_all = preprocessing(df)
data_length = len(cleaned_main_data_all)

word_dist_all = nltk.FreqDist(cleaned_main_data_all)
result_all = pd.DataFrame(word_dist_all.most_common(),
                          columns=['Word', 'Frequency'])
# Number of unique words in the whole database
beta = result_all.shape[0]

word_dist = nltk.FreqDist(cleaned_main_data_all)
rslt = pd.DataFrame(word_dist.most_common(),
                    columns=['Word', 'Frequency'])


counter = 0
for word in rslt['Word']:
    f.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}'.format(counter, word,
                               dictionary_story[word].frequency, dictionary_story[word].probability,
                               dictionary_ask_hn[word].frequency, dictionary_ask_hn[word].probability,
                               dictionary_show_hn[word].frequency, dictionary_show_hn[word].probability,
                               dictionary_poll[word].frequency, dictionary_poll[word].probability
                               ))
    f.write("\n")
    counter += 1



f.close()


def proc (text):

    a = text.lower()
    b = re.sub('[^A-Za-z]+', ' ', a)
    word_tokens = word_tokenize(b)
    lemmatizer = WordNetLemmatizer()
    cleaned_data_title = [word for word in word_tokens if not word.isnumeric()]
    data = [lemmatizer.lemmatize(word) for word in cleaned_data_title]
    #data_stop_words_filter = stop_words(data)
    #data_length_filter = [word for word in data if len(word) < 9 and len(word) > 2]

    return data


def NaiveByes():

    f = open("result.txt", "w+")
    counter = 0
    compare = ''
    right=0
    wrong =0

    for ind in df_test.index:

            sent = df_test['Title'][ind]
            _type = df_test['Post Type'][ind]

            classifier_res = {}
            sent_clean = proc(sent)

            sentence_story_prob = math.log(probability_story)
            sentence_ask_prob = math.log(probability_ask)
            sentence_show_prob = math.log(probability_show)
            sentence_poll_prob = math.log(probability_poll)

            for word in sent_clean:

                if cleaned_main_data_all.__contains__(word):

                   story_prob = dictionary_story[word].probability
                   ask_prob = dictionary_ask_hn[word].probability
                   show_prob = dictionary_show_hn[word].probability
                   poll_prob = dictionary_poll[word].probability

                else:
                    row_count = df.shape[0]
                    #beta = row_count * row_count
                    story_prob = ask_prob = show_prob = poll_prob = alpha/(data_length + beta * alpha)

                sentence_story_prob += math.log(story_prob)
                sentence_ask_prob += math.log(ask_prob)
                sentence_show_prob += math.log(show_prob)
                sentence_poll_prob += math.log(poll_prob)
                classifier_res[sentence_story_prob] = 'story'
                classifier_res[sentence_ask_prob] = 'ask_hn'
                classifier_res[sentence_show_prob] = 'show_hn'
                classifier_res[sentence_poll_prob] = 'poll'

                max_prob = max(sentence_story_prob, sentence_ask_prob, sentence_show_prob, sentence_poll_prob )
                result = classifier_res[max_prob]


                if(result == str(_type)):
                    compare = 'right'

                else:
                    compare = 'wrong'

            df_test.at[ind,'Compare'] = compare
            df_test.at[ind,'Result'] = result

            counter +=1



            f.write('{}  {}  {}  {}  {}  {}  {}  {}  {}'.format(counter, sent,
                                                          result,sentence_story_prob,
                                                          sentence_ask_prob,
                                                          sentence_show_prob,
                                                          sentence_poll_prob,
                                                          str(_type),
                                                          compare))
            f.write("\n")


    f.close()

#--------------Accuracy----------------
    true_type = df_test.loc[df_test['Compare'] == 'right']
    true_res = true_type['Compare'].value_counts()

    false_type = df_test.loc[df_test['Compare'] == 'wrong']
    false_res = false_type['Compare'].value_counts()

    print("Wrong: ", false_res)

    accuracy_all = true_res / len(df_test)

    print("accuracy: ", accuracy_all)


#--------------Precision and Recall-------------------

confusion_matrix = [[0 for j in range(4)] for i in range(4)]
    #       story  show  poll  ask ----> Actual Values
    # story   0     0     0     0
    # show    0     0     0     0
    # poll    0     0     0     0
    # ask     0     0     0     0

def prec_rec():

    story_right = 0
    show_right = 0
    ask_right = 0
    poll_right = 0

    story_type_df = df_test.loc[df_test['Post Type'] == 'story']
    story_count_val = story_type_df['Post Type'].value_counts()

    show_type_df = df_test.loc[df_test['Post Type'] == 'show_hn']
    show_count_val = show_type_df['Post Type'].value_counts()

    ask_type_df = df_test.loc[df_test['Post Type'] == 'ask_hn']
    ask_count_val = ask_type_df['Post Type'].value_counts()

    poll_type_df = df_test.loc[df_test['Post Type'] == 'poll']
    poll_count_val = poll_type_df['Post Type'].value_counts()



    story_lock = df_test.loc[df_test['Post Type'] == 'story']

    for ind in story_lock.index:
        if story_lock['Result'][ind] == 'story':
            confusion_matrix[0][0] += 1
        if story_lock['Result'][ind] == 'show_hn':
            confusion_matrix[1][0] += 1
        if story_lock['Result'][ind] == 'poll':
            confusion_matrix[2][0] += 1
        if story_lock['Result'][ind] == 'ask_hn':
            confusion_matrix[3][0] += 1

        if story_lock['Compare'][ind] == 'right':
            story_right += 1


    show_lock = df_test.loc[df_test['Post Type'] == 'show_hn']

    for ind in show_lock.index:
        if show_lock['Result'][ind] == 'story':
            confusion_matrix[0][1] += 1
        if show_lock['Result'][ind] == 'show_hn':
            confusion_matrix[1][1] += 1
        if show_lock['Result'][ind] == 'poll':
            confusion_matrix[2][1] += 1
        if show_lock['Result'][ind] == 'ask_hn':
            confusion_matrix[3][1] += 1

        if show_lock['Compare'][ind] == 'right':
            show_right += 1

    poll_lock = df_test.loc[df_test['Post Type'] == 'poll']

    for ind in poll_lock.index:
        if poll_lock['Result'][ind] == 'story':
            confusion_matrix[0][2] += 1
        if poll_lock['Result'][ind] == 'show_hn':
            confusion_matrix[1][2] += 1
        if poll_lock['Result'][ind] == 'poll':
            confusion_matrix[2][2] += 1
        if poll_lock['Result'][ind] == 'ask_hn':
            confusion_matrix[3][2] += 1

        if poll_lock['Compare'][ind] == 'right':
            poll_right += 1

    ask_lock = df_test.loc[df_test['Post Type'] == 'ask_hn']

    for ind in ask_lock.index:
        if ask_lock['Result'][ind] == 'story':
            confusion_matrix[0][3] += 1
        if ask_lock['Result'][ind] == 'show_hn':
            confusion_matrix[1][3] += 1
        if ask_lock['Result'][ind] == 'poll':
            confusion_matrix[2][3] += 1
        if ask_lock['Result'][ind] == 'ask_hn':
            confusion_matrix[3][3] += 1

        if ask_lock['Compare'][ind] == 'right':
            ask_right += 1

    # Precision = TP / TP + FP
    # Recall = TP/ TP + FN

    TP_all = confusion_matrix[0][0] + confusion_matrix [1][1] + confusion_matrix[2][2] + confusion_matrix [3][3]

    TP_story = confusion_matrix[0][0]
    TP_show = confusion_matrix [1][1]
    TP_poll = confusion_matrix[2][2]
    TP_ask = confusion_matrix[3][3]

    FP_stroy = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]
    FP_show = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]
    FP_poll = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]
    FP_ask = confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    FN_stroy = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]
    FN_show = confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]
    FN_poll = confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]
    FN_ask = confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]

    precision_story = TP_story / (TP_story + FP_stroy)
    precision_show = TP_show / (TP_show + FP_show)

    if TP_poll>0 or FP_poll >0:
        precision_poll = TP_poll / (TP_poll + FP_poll)
    else:
        precision_poll = 0

    precision_ask = TP_ask / (TP_ask + FP_ask)

    recall_story = TP_story / (TP_story + FN_stroy)
    recall_show = TP_show / (TP_show + FN_show)

    if TP_poll>0 or FN_poll >0:
        recall_poll = TP_poll / (TP_poll + FN_poll)
    else:
        recall_poll = 0

    recall_ask = TP_ask / (TP_ask + FN_ask)

    average_precision = (precision_ask + precision_poll + precision_show + precision_story) /4
    average_recall = (recall_ask + recall_poll + recall_show + recall_story) / 4

    F1 = 2 * (( average_precision * average_recall) / (average_precision + average_recall))

    print('------------------------------------')

    print("*Total Precision: ", average_precision)
    print("* Total Recall: ", average_recall)
    print("* Total F1 Score:", F1)
    print('------------------------------------')

    print("*story Precision: ", precision_story)
    print("*story Recall: ", recall_story)
    print("*story Accuracy:", story_right/story_count_val)

    print('------------------------------------')
    print("*show_hn Precision: ", precision_show)
    print("*show_hn Recall: ", recall_show)
    print("*show_hn Accuracy:", show_right / show_count_val)
    print('------------------------------------')
    print("*ask_hn Precision: ", precision_ask)
    print("*ask_hn Recall: ", recall_ask)
    print("*ask_hn Accuracy:", ask_right / ask_count_val)
    print('------------------------------------')
    print("*poll Precision: ", precision_poll)
    print("*poll Recall: ", recall_poll)
    print("*poll Accuracy:", poll_right / poll_count_val)
    print('------------------------------------')

#----------------------RUN HERE ---------------------------

NaiveByes()

print("Alpha Value:", alpha)
prec_rec()
print(confusion_matrix)




