import pandas as pd
import re
import nltk


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    """
    Pre-process a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmatization is to be applied
    :return
        cleaned text
    """

    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = text.split()
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)

    return text


# Load training data --------------------------------------------------
with open('Data\DBLPTrainset.txt') as f:
    lines = f.readlines()

# Get only the label and the sentence and store them
for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [pre[0], ' '.join(pre[1:])]

# Store in pandas data frame
train = pd.DataFrame(lines, columns=['Label', 'Title'])

# Load test data------------------------------------------------------
with open('Data\DBLPTestset.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [' '.join(pre[1:])]

test = pd.DataFrame(lines, columns=['Title'])

with open('Data\DBLPTestGroundTruth.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()[1:][0]
test['Label'] = lines

# Pre-processing ----------------------------------------------------
lst_stopwords = nltk.corpus.stopwords.words("english")
train["Title_clean"] = train["Title"].apply(
    lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))
test["Title_clean"] = test["Title"].apply(
    lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True))

# Store results -----------------------------------------------------
X_train = train.Title_clean
X_test = test.Title_clean
y_train = train.Label
y_test = test.Label


