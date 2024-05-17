import nltk

# Проверка установки необходимых пакетов
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
