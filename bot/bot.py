# -- coding: utf-8 -
from telegram.ext  import Updater, CommandHandler, MessageHandler, Filters
import pickle
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import  FastText,KeyedVectors
import pickle
import numpy as np
import time


patch_ft = r'D:\train\Otvety/ft.bin'
patch_ft_index = r'D:\train\Otvety/ft_index+full'
patch_index_map = r'D:\train\Otvety/index_map_new.pkl'

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)


def get_data(patch_ft,patch_ft_index,patch_index_map):
    modelFT=KeyedVectors.load_word2vec_format(patch_ft,binary=False)
    ft_index = annoy.AnnoyIndex(300 ,'angular')
    ft_index.load(patch_ft_index)
    with open(patch_index_map, 'rb') as f:
        index_map = pickle.load(f)
    return modelFT,index_map,ft_index

modelFT,index_map,ft_index = get_data(patch_ft,patch_ft_index,patch_index_map)
print('Данные загружены')


#Настройки
updater = Updater(token='****') # Токен API к Telegram
dispatcher = updater.dispatcher

hello_text = "Привет, я бот изобретенный Виталиком. \n  \
             Я не очень умный бот, поэтому не обижайтесь. \n \
             Учился я отвечать на вопросы на просторах интернета, а именно на сервисе 'Вопроы на mail.ru'"



def preprocess_txt(line,exclude,morpher,sw):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls

def get_respons(question, index, model, index_map):
    question = preprocess_txt(question,exclude,morpher,sw)
    vector = np.zeros(300)
    norm = 0
    for word in question:
        if word in model:
            vector += model[word]
            norm += 1
    if norm > 0:
        vector = vector / norm
    answers = index.get_nns_by_vector(vector, 1)
    return np.random.choice(index_map[answers[0]])

# Обработка команд
def startCommand(update,colback):
    colback.bot.send_message(chat_id=update.message.chat_id, text=hello_text)
    print(time.strftime("%H:%M:%S"))
    print(f'start чат {update.message.chat_id}')

def textMessage(update,colback):
    text =  update.message.text #это текст запроса который можно обрабатывать классифицировать для формирования ответа
    answer = get_respons(text, ft_index, modelFT, index_map)
    colback.bot.send_message(chat_id=update.message.chat_id, text=answer)
    print(time.strftime("%H:%M:%S"))
    print(f' чат {update.message.chat_id} сообщение {text}')
    print(f' ответ {answer}')
# Хендлеры
start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
# Добавляем хендлеры в диспетчер
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
# Начинаем поиск обновлений
updater.start_polling(drop_pending_updates=True)
# Останавливаем бота, если были нажаты Ctrl + C
updater.idle()