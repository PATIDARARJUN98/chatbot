from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import os

chatbot = ChatBot('Bot')
trainer = ChatterBotCorpusTrainer(chatbot)

corpus_path = 'C:/Users/thispc/Downloads/chatterbot-corpus-master/chatterbot_corpus/data/english/'

for file in os.listdir(corpus_path):
     trainer.train(corpus_path + file)

while True:
        message = input('You:')
        if message.strip() != 'Bye':
                reply = chatbot.get_response(message)
                print('ChatBot :' ,reply)
        if message.strip() =='Bye':
                print('ChatBot : Bye')
                break