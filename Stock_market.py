
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fastText
# Read data
df = pd.read_csv('data/Combined_News_DJIA.csv', parse_dates=True, index_col=0)

print(df.columns)
# Select only the top N news to use as features of the classifier. N ranges from 1 to 25.
# In this case, N = 20. 
N = [15,20,25]
dic_list = {}
for n in N:
    columns = ['Top' + str(i+1) for i in range(n)]
    print(columns)

    df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # df_reversed =  df.iloc[::-1]

    # d = []
    # for i in range(0,len(df_reversed)):
    #     d1 = (df_reversed.iloc[i:i+3])
    #     # print(len(d1['joined'].values))
    #     # print(d1)
    #     d.append(' '.join(d1['joined'].values))
     

    # df2 = pd.DataFrame(columns = ['Date','joined','Label'])
    # df2['Date'] = df_reversed.index
    # df2['joined'] = d
    # df2['Label'] = df_reversed['Label'].values

    # print(df2.head())
    # # Create a new dataframe with only Label and joined columns
    df1 = df[['Label', 'joined']].copy()
    print(df1.head())
    from sklearn.model_selection import train_test_split

    # x_train, x_test, y_train, y_test = train_test_split(df1['joined'],df1['Label'], test_size = 0.20, shuffle = False)

    # print(x_train.head())
    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
    }

    def clean_text(text, remove_stopwords = False):
        '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
        
        # Convert words to lower case
        text = text.lower()
        
        # Replace contractions with their longer forms 
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)
        
        # Format words and remove unwanted characters
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'0,0', '00', text) 
        text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'\$', ' $ ', text)
        text = re.sub(r'u s ', ' united states ', text)
        text = re.sub(r'u n ', ' united nations ', text)
        text = re.sub(r'u k ', ' united kingdom ', text)
        text = re.sub(r'j k ', ' jk ', text)
        text = re.sub(r' s ', ' ', text)
        text = re.sub(r' yr ', ' year ', text)
        text = re.sub(r' l g b t ', ' lgbt ', text)
        text = re.sub(r'0km ', '0 km ', text)
        
        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text

    # Clean the headlines
    import re
    clean_headlines = []
    from nltk.corpus import stopwords
    for daily_headlines in df1['joined']:
    #     clean_daily_headlines = []
    #     for headline in daily_headlines:
    #         clean_daily_headlines.append(clean_text(headline))
        clean_headlines.append(clean_text(daily_headlines))

    data = pd.DataFrame(columns = ['headlines'])
    data['headlines'] = clean_headlines
    data.to_csv('data.csv','w')
    print(clean_headlines[0])
    # In[2]:


    # Find the number of times each word was used and the size of the vocabulary
    word_counts = {}

    for date in clean_headlines:
    #     print(date)
    #     for headline in date:
        for word in date.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    #     break
                
    print("Size of Vocabulary:", len(word_counts))
    nb_words = len(word_counts)


    # In[3]:


    # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe
    threshold = 1
    #dictionary to convert words to integers
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold :
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    print("Total Number of Unique Words:", len(word_counts))
    print("Number of Words we will use:", len(vocab_to_int))
    print("Percent of Words we will use: {}%".format(usage_ratio))

    print('loading embedding')
    embedding_matrix = np.zeros((len(vocab_to_int), 300))

    ft_en = fastText.load_model("../Project/summarization/Word_Embeddings/Fasttext/wiki.en.bin")

    for i,word in enumerate(vocab_to_int):
        try:
            embedding_vector = ft_en.get_word_vector(word.lower())
        except:
            embedding_vector = np.asarray([0]*300)
        embedding_matrix[i] = embedding_vector

    print('done')


    # Change the text from words to integers
    # If word is not in vocab, replace it with <UNK> (unknown)
    print('changing sentences to integer')
    word_count = 0
    unk_count = 0
    print('total no of samples',len(df1['Label']))
    int_headlines=[]
    for i,headline in enumerate(clean_headlines):
        #print(headline)
        int_headline = []
        for word in headline.split():
            word_count += 1
            if word in vocab_to_int:
                int_headline.append(vocab_to_int[word])
            else:
                int_headline.append(vocab_to_int["<UNK>"])
                unk_count += 1
        int_headlines.append(int_headline)


    unk_percent = round(unk_count/word_count,4)*100

    print("Total number of words in headlines:", word_count)
    print("Total number of UNKs in headlines:", unk_count)
    print("Percent of words that are UNK: {}%".format(unk_percent))

    print(int_headlines[0:10])
    # In[5]:
    print('loading')
    import pickle 
    with open('data_init.pickle', 'wb') as fp:
         pickle.dump(int_headlines,fp)

    # with open('data.pickle', 'rb') as fp:
    #      pad_headlines = pickle.load(fp)


    # In[4]:


    # Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
    # These values are chosen to not have an excessively long training time and 
    # balance the number of headlines used and the number of words from each headline.
    # Find the length of headlines
    lengths = []
    for headline in int_headlines:
            lengths.append(len(headline))

    # Create a dataframe so that the values can be inspected
    lengths = pd.DataFrame(lengths, columns=['counts'])
    print(lengths.describe())

    print('padding')
    max_daily_length = 200
    pad_headlines = []
    for headline in int_headlines:
    #     print(headline)
        # Pad daily_headlines if they are less than max length
        if len(headline) < max_daily_length:
            for i in range(max_daily_length-len(headline)):
                pad = vocab_to_int["<PAD>"]
                headline.append(pad)
        # Limit daily_headlines if they are more than max length
        else:
            headline = headline[:max_daily_length]
    #     print(len(headline))
        pad_headlines.append(headline)
    print(pad_headlines[0:10])


    # # In[5]:
    # print('loading')
    # import pickle 
    # # with open('data.pickle', 'wb') as fp:
    # #      pickle.dump(pad_headlines,fp)

    # with open('data.pickle', 'rb') as fp:
    #      pad_headlines = pickle.load(fp)

    print(len(pad_headlines))
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(pad_headlines, df1['Label'], test_size = 0.20, shuffle = False)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    # In[6]:


    from keras.models import Sequential,Model
    from keras.layers.wrappers import Bidirectional
    from keras.layers import Input, RepeatVector, concatenate, Activation, Permute, merge,Reshape
    from keras.layers import Embedding,Conv1D,TimeDistributed,MaxPooling1D,LSTM,Dense,Dropout
    from keras.optimizers import Adam

    # In[7]:


    from keras.utils import to_categorical
    train_sentences = (x_train)
    train_sentences = train_sentences.reshape(train_sentences.shape[0],train_sentences.shape[1])
    train_class = y_train
    train_class = to_categorical(train_class,2)


    # In[8]:


    test_sentences = (x_test)
    test_sentences = test_sentences.reshape(test_sentences.shape[0],test_sentences.shape[1])
    test_class = y_test
    test_class = to_categorical(test_class,2)


    # In[9]:


    def class_only(seq_len, emb_dim, emb,vocab_size,label_size):
        model = Sequential()
        model.add(Embedding(vocab_size, emb_dim, weights=[emb],trainable=False))
        model.add(Conv1D(100, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add((LSTM(100)))
        # model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # optimizer = Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    # def class_only(seq_len, emb_dim, emb,vocab_size,label_size):

    #     inputs = Input(name='inputs',shape=[seq_len])
    #     layer = Embedding(vocab_size, emb_dim, weights=[emb],trainable=False)(inputs)
    #     layer = LSTM(64)(layer)
    #     layer = Dense(256,name='FC1')(layer)
    #     layer = Activation('relu')(layer)
    #     layer = Dropout(0.5)(layer)
    #     layer = Dense(1,name='out_layer')(layer)
    #     layer = Activation('sigmoid')(layer)
    #     model = Model(inputs=inputs,outputs=layer)
    #     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #     return model

    # def class_only(seq_len, emb_dim, emb,vocab_size,label_size):
    #     ip = Input(shape=(seq_len,))
    #     e = Embedding(vocab_size, emb_dim, weights=[emb],trainable=False)(ip)
    #     e=(Reshape((seq_len,emb_dim)))(e)

    #     ####CNN####
    #     c=Conv1D(100,kernel_size=2,padding='same',activation='relu')(e)
    #     c=Dropout(0.3)(c)
    #     c1=Conv1D(100,kernel_size=3,padding='same',activation='relu')(c)
    #     c1=Dropout(0.3)(c1)

    #     ###BiLSTM###
    #     l = (Bidirectional(LSTM(75)))(c1)
        
    #     # merge=concatenate([l,c,c1])
    #     out=Dense(100, activation='relu')(l)
    #     out=Dropout(0.3)(out)
    #     out=Dense(75, activation='relu')(out)
    #     out=Dropout(0.3)(out)
          
    #     out=Dense(label_size, activation='softmax')(out)
        
    #     model = Model(inputs=ip, outputs=out)
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model

    # In[10]:


    type(x_train)


    # In[11]:


    label_size = 2
    seq_len = 200
    vocab_size = len(vocab_to_int)
    emb_dim = 300
    model = class_only(seq_len, emb_dim, embedding_matrix, vocab_size, label_size) 
    print(model.summary())


    # In[12]:


    print(train_class.shape)


    # In[ ]:


    epochs = 50
    from keras.models import load_model
    from keras import callbacks
    checkpoint_path = 'Model/' + str(n) + '_final_model_' + str(epochs) + '.hdf5'
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_loss',
                                save_best_only=True, save_weights_only=False, mode='auto', period=1)

    history = model.fit(train_sentences,train_class, validation_data=(test_sentences,test_class), epochs=epochs, batch_size=32,verbose=2,callbacks=[checkpoint])

    print("finished training...")
    model = load_model(checkpoint_path)


    # In[ ]:


    class_predict = model.predict(test_sentences)

    print(class_predict)
    class_predict = np.argmax(class_predict,axis =1)


    # In[ ]:

    from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score,roc_auc_score,classification_report,precision_recall_fscore_support,accuracy_score
    print(class_predict)
    print(classification_report)
    print(y_test)
    print("Class",classification_report(y_test,class_predict))
    print('accuracy',accuracy_score(y_test,class_predict))

    print(history.history.keys())
    # summarize history for accuracy
    plt.switch_backend('agg')
    fig_1 = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    save_path1 = 'Figures/1st_' + str(n) + '_' + str(epochs)
    fig_1.savefig(save_path1, bbox_inches='tight')
    plt.show()
    # summarize history for loss
    fig_2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    save_path2 = 'Figures/2nd_' + str(n) + '_' + str(epochs)

    fig_2.savefig(save_path2, bbox_inches='tight')
    plt.show()

    dic_list[n] = accuracy_score(y_test,class_predict)


    with open(save_path1 + '.txt', "w") as text_file:
         print("Classification report: {}".format(classification_report(y_test,class_predict)), file=text_file)

    

with open(save_path2 + '.txt', "w") as text_file:
     print("Classification report: {}".format(dic_list), file=text_file)

print(dic_list)
