from telegram.ext  import Updater, CommandHandler, MessageHandler, Filters
import numpy as np
import time
import re
import torch
from transformers import BertModel, BertTokenizerFast
import annoy


path_answers = r'D:\train\Otvety\bert/answers.npy'
patch_index = r'D:\train\Otvety\bert/index_full'

bert_index = annoy.AnnoyIndex(768 , 'angular')
bert_index.load(patch_index)
answers = np.load(path_answers,allow_pickle=True,encoding='bytes')
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

print('Данные загружены')

#Настройки
updater = Updater(token='1975804308:AAEV02-2kVTTHm0zy9ijr7Pq5eISA9SHY4c') # Токен API к Telegram
dispatcher = updater.dispatcher

hello_text = "Привет, я бот. \n  \
             Я не очень умный бот, поэтому не обижайтесь. \n \
             Учился я отвечать на вопросы на просторах интернета, а именно на сервисе 'Вопроы на mail.ru использую BERT'"


def clean_text_questions(text):
    text = text.lower()
    text = re.sub(r'[^\w+]',' ',text)
    text = re.sub(r' +',' ',text)
    return text

def get_respons(question):
    question = clean_text_questions(question)
    question = question[:400]
    tok = tokenizer(question, return_token_type_ids=False, return_tensors='pt')
    with torch.no_grad():
        vector = model(**tok)[1].numpy()[0]
    answer_ind = bert_index.get_nns_by_vector(vector, 1)
    my_answers = answers[answer_ind][0]
    return str(np.random.choice(my_answers))

# Обработка команд
def startCommand(update,colback):
    colback.bot.send_message(chat_id=update.message.chat_id, text=hello_text)
    print(time.strftime("%H:%M:%S"))
    print(f'start чат {update.message.chat_id}')

def textMessage(update,colback):
    text =  update.message.text #это текст запроса который можно обрабатывать классифицировать для формирования ответа
    answer = get_respons(text)
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