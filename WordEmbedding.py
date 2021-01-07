import gensim
import numpy as np
from keras import models, layers, preprocessing as kprocessing


def lstm(X_train, X_test, y_train):
    """"
    Method that performs text classification by using a neural network model containing an LSTM
    It does so by the following steps:
    1) Creating an embedding model based on the Word2Vec framework
    2) Using this embedding as embedding layer in a deep NN containing an LSTM and softmax output
    """

    X_train_lst = [string.split() for string in X_train]
    X_test_lst = [string.split() for string in X_test]

    # tokenize text
    tokenizer = kprocessing.text.Tokenizer(split=' ', oov_token="NaN")
    tokenizer.fit_on_texts(X_train_lst)  # apply the tokenizer and store the dictionary
    dic_vocabulary = tokenizer.word_index

    # create sequence of integers for each text
    lst_text2seq_train = tokenizer.texts_to_sequences(X_train_lst)
    lst_text2seq_test = tokenizer.texts_to_sequences(X_test_lst)
    # padding sequence so that all sequences have equal length
    seq_len = max([len(seq) for seq in lst_text2seq_train])

    X_train = kprocessing.sequence.pad_sequences(lst_text2seq_train, maxlen=seq_len, padding="post", truncating="post")
    X_test = kprocessing.sequence.pad_sequences(lst_text2seq_test, maxlen=seq_len, padding="post", truncating="post")

    emb_size = 100
    emb = gensim.models.word2vec.Word2Vec(X_train_lst, size=emb_size, window=8, min_count=1, sg=1, iter=30)
    embeddings = np.zeros((len(dic_vocabulary) + 1, emb_size))
    # Every word in the learned dictionary gets one row where its embedding gets
    for word, idx in dic_vocabulary.items():
        # update the row with vector
        try:
            embeddings[idx] = emb[word]
        # if word not in the embedding model then skip and the row stays all 0s
        except:
            pass

    # encode y
    dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])

    # train
    model = NeuralNet(seq_len, embeddings)
    model.fit(x=X_train, y=y_train, batch_size=64, epochs=20, shuffle=True, verbose=1, validation_split=0.3)

    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
    return predicted


def NeuralNet(seq_len, embeddings):
    # input
    x_in = layers.Input(shape=(seq_len,))
    # embedding
    x = layers.Embedding(input_dim=embeddings.shape[0],  # number of embeddings
                         output_dim=embeddings.shape[1],  # size of embedding vector
                         weights=[embeddings],
                         input_length=seq_len, trainable=False)(x_in)

    # 2 layers of bidirectional lstm + one dense layer with ReLU activ. and softmax output
    x = layers.Bidirectional(layers.LSTM(units=seq_len, dropout=0.3, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=seq_len, dropout=0.3))(x)
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(5, activation='softmax')(x)  # 5 is number of classes (conferences)

    # compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
