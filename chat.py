import pandas as pd
import nltk
import words as words
nltk.download('punkt')
from nltk.corpus import stopwords
ensw = stopwords.words('english')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tflearn
import tensorflow
import random

import json
with open('vendor.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
words = [stemmer.stem(w.lower()) for w in words if w != "?"] #convert all cases in small later and convert the words into roots or base form and remove unwanted character
words = sorted(list(set(words))) #remove the duplicate words


# labels = sorted(labels) #remove the duplicate classes
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)
filterArr = [[item for item in doc if item not in ensw] for doc in docs_x]
print(filterArr)





training = numpy.array(training)
output = numpy.array(output)


tensorflow.reset_default_graph() # resetting underline graph data

net = tflearn.input_data(shape=[None, len(training[0])]) #building neural network (1)input layer to feeded the data to the neural network
net = tflearn.fully_connected(net, 8) #these two layer are hidden layer both have 10 nodes and edges
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer have no. of classes depect by len(output[0]), activation function is used softmax
net = tflearn.regression(net) #its an regression layer is used in tflearn for apply regression for provided input

model = tflearn.DNN(net) # dnn is an model rapper that can automatically performed a neural network classifer task such as prediction, training save

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #fit model with training data
model.save("model.tflearn")

# def conv(u,b):
#     dfm = pd.read_excel("conv_logs.xlsx", "Sheet1")
#     #dfm = pd.read_csv("conv_logs.csv")
#     df1=pd.Series([u,b],index=['user','bot'])
#     dfm=dfm.append(df1,ignore_index=True)
#     #print(dfm)
#     #dfm.to_csv('conv_logs.csv',header=True)
#     dfm.to_excel("conv_logs.xlsx", sheet_name="Sheet1")



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]


        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['response']
                dfm = pd.read_excel("conv_logs.xlsx", "Sheet1")
                # dfm = pd.read_csv("conv_logs.csv")
                df1 = pd.Series([inp, tag], index=['user', 'bot'])
                dfm = dfm.append(df1, ignore_index=True)
                # print(dfm)
                # dfm.to_csv('conv_logs.csv',header=True)
                dfm.to_excel("conv_logs.xlsx", sheet_name="Sheet1",index = False)

        print(random.choice(responses))


        # conv(inp, tag)

chat()

