import re
import string
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


def convertCamelCase(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    finals = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    #     print(finals)
    return finals


def text_to_wordlist(text, remove_stopwords=True, stem_words=False, remove_punc=True):
    # URL remove
    #     print(type(text))
    text = str(text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    # split words:
    text = text.split()
    text2 = []
    for t in text:
        splited = re.split("[.,' \-!?:#^%*[$\]+|_`\)=<;{\"&>@/~(\\}\\\]+", t)
        text2.extend(splited)
    not_digits = []
    for i in text2:
        if i.isdigit():
            not_digits.append("cc")
        else:
            not_digits.append(i)
    text4 = []
    for c in not_digits:
        if len(c) > 1:
            if c[1].islower():
                for k in c[1:]:
                    if not k.islower():
                        text4.append(convertCamelCase(c))
                        break;
                else:
                    text4.append(c)
            else:
                text4.append(c)
        else:
            text4.append(c)

    text5 = []
    for t in text4:
        splited = re.split("_", t)
        text5.extend(splited)
    text6 = []
    for x in text5:
        if len(x) > 1:
            text6.append(x)

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text6 if not w in stops]
    # print("after stop ", text)
    # print("after stop word", text)
    text = " ".join(text)

    text = re.sub("  ", " ", text)
    text = re.sub("   ", " ", text)
    text = re.sub("    ", " ", text)
    text = re.sub("[0-9]+", "CC", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"couldn't", "could not ", text)
    text = re.sub(r"doesn’t", "does not ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"“", "", text)
    text = re.sub(r"”", "", text)

    if remove_punc:
        exclude = set(string.punctuation)
        text = ''.join(ch for ch in text if ch not in exclude)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    text = text.lower()

    return (text)

def getData(type_str):
    assert type_str in ['train', 'test']
    data = pd.read_csv(open('data/medium_link_prediction_noClue_shuffled_'+ type_str +'.csv', 'r', encoding='utf-8'))
    if type_str == 'train':
        # delete train_data invalid columns
        data = data.drop(index=[3905,3906,
                                5763,5764,
                                8271,8272,
                                13407,13408,13409,13410,13411,13412,13413,13414,13415,13416,
                                15692,15693,
                                16946,16947,16948,16949,16950,16951,16952,16953,16954,16955,
                                18452,18453,
                                19274,19275,19276,
                                21563,21564,21565,21566,
                                24173,24174,24175,24176,
                                24819,24820,24821,24822,24823,24824,
                                24988,24989,24990,24991,
                                25464,25465,25466,25467,25468,
                                25495,25496,
                                27510,27511,
                                30827,30828,30829
                                ])
    elif type_str == 'test':
        # delete test_data invalid columns
        data = data.drop(index=[1801,1802])
    ids = []
    sentences = []
    index_list = []
    map_relation = {'duplicate':0,'direct':1,'indirect':2,'isolated':3}
    for row in data.itertuples(index=False):
        id_list = []
        sentence_list = []
        id = getattr(row, 'id')
        print(eval(id))
        class_type = map_relation[getattr(row, '_23').strip()]
        q1_id = getattr(row, 'q1_Id')
        q1_title = getattr(row, 'q1_Title')
        q1_body = getattr(row, 'q1_Body')
        if q1_body is np.nan:
            q1_sentence = q1_title.strip()
        else:
            q1_sentence = q1_title.strip() + ' ' + q1_body.strip()
        id_list.append(eval(q1_id))
        sentence_list.append(q1_sentence)
        q1_answersIdlist = getattr(row, 'q1_AnswersIdList')
        q1_answersBody = getattr(row, 'q1_AnswersBody')
        if not q1_answersIdlist == r'\N' and not q1_answersBody == r'\N':
            id_list += eval(q1_answersIdlist)
            sentence_list += eval(q1_answersBody)

        q2_id = getattr(row, 'q2_Id')
        q2_title = getattr(row, 'q2_Title')
        q2_body = getattr(row, 'q2_Body')
        if q2_body is np.nan:
            q2_sentence = q2_title.strip()
        else:
            q2_sentence = q2_title.strip() + ' ' + q2_body.strip()
        id_list.append(eval(q2_id))
        sentence_list.append(q2_sentence)
        q2_answersIdlist = getattr(row, 'q2_AnswersIdList')
        q2_answersBody = getattr(row, 'q2_AnswersBody')
        if not q2_answersIdlist == r'\N' and not q2_answersBody == r'\N':
            id_list += eval(q2_answersIdlist)
            sentence_list += eval(q2_answersBody)
        for t_id, t_sentence in zip(id_list,sentence_list):
            if t_id not in ids:
                ids.append(t_id)
                sentences.append(t_sentence)

        index_list.append([eval(id),eval(q1_id),eval(q2_id),class_type])

    index_data = np.array(index_list)
    index_data = pd.DataFrame(index_data, columns=['id', 'q1_id', 'q2_id', 'class'])
    index_data.to_csv('data/index_' + type_str + '_data.csv', index=False)


    new_data = pd.DataFrame(list(zip(ids,sentences)), columns=['id', 'sentence'])
    new_data.to_csv('data/origin_' + type_str + '_data.csv', index=False)

    new_data['sentence'] = new_data['sentence'].apply(text_to_wordlist,args=(True,False,True))
    new_data.to_csv('data/processed_' + type_str + '_data.csv', index=False)
