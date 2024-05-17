import cloudpickle
import spacy
import os


def load_model(filepath):
    with open(filepath, 'rb') as file:
        return cloudpickle.load(file)


# Определение базового пути
base_path = os.path.abspath(os.path.dirname(__file__))

# Загрузка всех необходимых моделей при старте приложения
task_vectorizer = load_model(os.path.join(base_path, '../static/models/task_vectorizer.pkl'))
task_classifier = load_model(os.path.join(base_path, '../static/models/task_classifier.pkl'))
stopwords = load_model(os.path.join(base_path, '../static/stopwords/russian_stopwords.pkl'))
technology_extractor = spacy.load(os.path.join(base_path, "../static/models/ru_core_news_technologies_md"))
