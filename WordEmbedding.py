import gensim
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import models, layers, preprocessing as kprocessing


def embedding(X_train, X_test, y_train, y_test):
    X_train_lst = [string.split() for string in X_train]

    # TODO: uitzoeken wat dit hieronder allemaal betekent
    emb = gensim.models.word2vec.Word2Vec(X_train_lst, size=300, window=8, min_count=1, sg=1, iter=30)
    # tokenize text
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                           oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(X_train_lst) # apply the tokenizer and store the dictionary
    dic_vocabulary = tokenizer.word_index
    # create sequence
    lst_text2seq = tokenizer.texts_to_sequences(X_train_lst)
    # padding sequence
    seq_len = 15
    X_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=seq_len, padding="post", truncating="post")

    X_test_lst = [string.split() for string in X_test]
    # text to sequence with the fitted tokenizer
    lst_text2seq = tokenizer.texts_to_sequences(X_test_lst)

    ## padding sequence
    X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=seq_len,
                                                padding="post", truncating="post")

    embeddings = np.zeros((len(dic_vocabulary) + 1, 300))
    # Every word in the learned dictionary gets one row where its embedding gets
    for word, idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] = emb[word]
        ## if word not in the embedding model then skip and the row stays all 0s
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
    x_in = layers.Input(shape=(seq_len,))
    ## embedding
    x = layers.Embedding(input_dim=embeddings.shape[0],
                         output_dim=embeddings.shape[1],
                         weights=[embeddings],
                         input_length=seq_len, trainable=False)(x_in)
    ## apply attention
    x = attention_layer(x, neurons=seq_len)
    ## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=seq_len, dropout=0.2,
                                         return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=seq_len, dropout=0.2))(x)
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
                         epochs=20, shuffle=True, verbose=1,
                         validation_split=0.3)

    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                 predicted_prob]

    # Test results ---------------------------------------
    # TODO: labels staan niet op goede volgorde!
    # Probleem: we gebruiken xticklabels = categories wat de unieke categories uit y_test zijn
    # Blijkbaar gebruik de cm een andere volgorde
    categories = y_test.unique()
    print("y_test:", y_test)
    print("predicted:", predicted)
    cm = confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=categories,
           yticklabels=categories, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.show()
