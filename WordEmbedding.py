import gensim
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import models, layers, preprocessing as kprocessing


def embedding(X_train, X_test, y_train, y_test):
    corpus = X_train

    ## create list of lists of unigrams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_corpus.append(lst_words)

    # TODO: uitzoeken wat dit hieronder allemaal betekent
    emb = gensim.models.word2vec.Word2Vec(lst_corpus, size=300, window=8, min_count=1, sg=1, iter=30)
    ## tokenize text
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                           oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index
    ## create sequence
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
    ## padding sequence
    X_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")

    corpus = X_test

    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_corpus.append(lst_words)

    ## text to sequence with the fitted tokenizer
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    ## padding sequence
    X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
                                                padding="post", truncating="post")

    embeddings = np.zeros((len(dic_vocabulary) + 1, 300))
    for word, idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] = emb[word]
        ## if word not in model then skip and the row stays all 0s
        except:
            pass

    ## code attention layer
    def attention_layer(inputs, neurons):
        x = layers.Permute((2, 1))(inputs)
        x = layers.Dense(neurons, activation="softmax")(x)
        x = layers.Permute((2, 1), name="attention")(x)
        x = layers.multiply([inputs, x])
        return x

    ## input
    x_in = layers.Input(shape=(15,))
    ## embedding
    x = layers.Embedding(input_dim=embeddings.shape[0],
                         output_dim=embeddings.shape[1],
                         weights=[embeddings],
                         input_length=15, trainable=False)(x_in)
    ## apply attention
    x = attention_layer(x, neurons=15)
    ## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2,
                                         return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
    ## final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(5, activation='softmax')(x)
    ## compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    ## encode y
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])

    ## train
    training = model.fit(x=X_train, y=y_train, batch_size=256,
                         epochs=50, shuffle=True, verbose=1,
                         validation_split=0.3)

    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                 predicted_prob]

    # Test results ---------------------------------------
    # TODO: labels staan niet op goede volgorde!
    categories = y_test.unique()
    cm = confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=categories,
           yticklabels=categories, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.show()
