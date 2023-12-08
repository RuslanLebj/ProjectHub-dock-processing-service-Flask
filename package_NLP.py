import pandas as pd
import pickle
import cloudpickle
import docx2txt
import re
import json
from string import punctuation
import pymorphy2
import nltk
from natasha import (
    Segmenter,
    MorphVocab,
    PER,
    NewsNERTagger,
    NewsEmbedding,
    Doc)
from yake import KeywordExtractor
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

morph = pymorphy2.MorphAnalyzer()
emb = NewsEmbedding()
segmenter = Segmenter()
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()


# Функция получения текста титульного листа
def get_title_page_text(text):
    # Слова окончания титульного листа
    title_page_stop_words = ['содержание', 'введение']
    for word in title_page_stop_words:
        text = text[:text.lower().find(word)]
    return text


# Функция предобрабтки для титульного листа
def title_page_preprocessing(text):
    pattern_removal = r'\([^()]*\)'
    # Паттерн для регулярного выражения для символов которые необходимо удалить
    text = re.sub(pattern_removal, r" ", text)
    pattern_required_chars = r'[^А-яёЁ.,0-9]'
    # Паттерн для регулярного выражения для символов которые необходимо оставить в тексте
    text = re.sub(pattern_required_chars, r' ', text)
    # Удалим последовательности пробелов
    text = re.sub(r" +", r" ", text)
    return text


# Получение ФИО групп "студент" и "преподаватель"
def get_group_names(text):
    # Токены(леммы слов), которые могут сбить NER модель и добавить слово в ФИО, так как нет знаков препинания
    stop_words_tokens = ['выполнил', 'доцент', 'защищен', 'ассистент']
    # Токены(леммы слов), которые обозначают студентов, выполнивших проект
    student_tokens = ['студент', 'обучающийся', 'студентка', 'обучающаяся', 'студенты', 'обучающиеся', ]
    # Токены(леммы слов), которые обозначают научных руководителей и руководителей практики по проекту
    leader_tokens = ['руководитель']
    # Список для сохранения имен
    names = []
    doc = Doc(text)
    # Деление на предложения и токены
    doc.segment(segmenter)
    # Отмечаем части предложения
    doc.tag_ner(ner_tagger)
    # Нормализуем
    for span in doc.spans:
        span.normalize(morph_vocab)
    # Выделяем все имена и добавляем в список
    for span in doc.spans:
        if span.type == PER:
            #  print(span)
            names += span.text.split(" ")
    # Удаляем повторяющиеся значения
    names = list(set(names))
    # Разделим инициалы, если таковые существуют
    preprocessed_names = []
    for name in names:
        tmp_names = name.split(".")
        preprocessed_names += tmp_names

    # Выделим имена студентов
    students_names = []
    is_student_bool = False
    is_student_counter = 0
    # Временный список для хранения фамилилии/имени/отчества
    full_name = []
    for token in doc.tokens:
        if is_student_bool and token.text in preprocessed_names and (not (token.text.lower() in stop_words_tokens)):
            full_name.append(token.text)
            is_student_counter += 1
            if is_student_counter == 3:
                is_student_bool = False
                is_student_counter = 0
                students_names.append(full_name)
        if token.text.lower() in student_tokens:
            is_student_bool = True
            is_student_counter = 0
            full_name = []

    # Выделим имена наунчных руководителей
    leaders_names = []
    is_leader_bool = False
    is_leader_counter = 0
    # Временный список для хранения фамилилии/имени/отчества
    full_name = []
    for token in doc.tokens:
        if is_leader_bool and token.text in preprocessed_names and (not (token.text.lower() in stop_words_tokens)):
            full_name.append(token.text)
            is_leader_counter += 1
            if is_leader_counter == 3:
                is_leader_bool = False
                is_leader_counter = 0
                leaders_names.append(full_name)
        if token.text.lower() in leader_tokens:
            is_leader_bool = True
            is_leader_counter = 0
            full_name = []

    return students_names, leaders_names


# Функция предобработки текста
def text_preprocessing(text):
    # Паттерн для регулярного выражения для заменяемых символов
    pattern_replacable_chars = r'[:;]'
    # Паттерн для регулярного выражения для символов на которые произойдут замены
    pattern_replaced_chars = r'.'
    # Паттерн для регулярного выражения для символов которые необходимо оставить тексте
    pattern_required_chars = rf'[^A-zА-я0-9Ёё \n\r{punctuation}]'
    # Заменяем символы в тексте
    text = re.sub(pattern_replacable_chars, pattern_replaced_chars, text)
    # Оставляем только необходимые символы
    text = re.sub(pattern_required_chars, r' ', text)
    return text


# Функция деления текста на строки с удалением пустых строк
def line_splitter(text):
    # Разделяем строки по символам перехода строк
    text_lines = re.split("\n|\t", text)
    # Удаляем пустые элементы
    text_lines = list(filter(len, text_lines))
    return text_lines


# Функция выделения определенной главы из текста
def chapter_selector(lines, chapter_number):
    chapter_started = False
    chapter_lines = []
    # Выделяем необходимую главу
    for line in lines:
        if not chapter_started:
            if f"глава {chapter_number}" in line.lower():
                chapter_started = True
        else:
            if "заключение" in line.lower():
                chapter_started = False
            elif f"глава {chapter_number + 1}" in line.lower():
                chapter_started = False
            chapter_lines.append(line)
    return chapter_lines


# Функция токенизации строк на предложения
def lines_to_sentences_tokenization(lines):
    sentences = []
    for line in lines:
        tokenized_line = nltk.sent_tokenize(line)
        sentences += tokenized_line
    return sentences


# Функция преодбработки предложений таблицы
def df_sentences_preprocessing(df_sentences):
    # Паттерн для регулярного выражения для символов и конструкций которые необходимо удалить
    pattern_removal = r'\([^()]*\)|\[[^()]*\]|\d+[.,]?\d+|[\d\)\(\]\[]|\.$'
    # | - знак соединения регулярных выражений
    # [^()]* - содержимое между
    # $ - конец строки
    # \d+ - последвательность чисел
    # [.,]? - . или ,
    # \([^()]*\) - круглые скобки и содержимое между ними
    # \[[^()]*\] - квадратные скобки и содержимое между ними
    # \d+[.,]?\d+ - дробные числа разделенные точкой или запятой
    # [\d\)\(\]\[] - числа, круглые скобки, квадратные скобки
    # \.$ - точка в конце предложения
    df_sentences["sentence"] = df_sentences["sentence"].map(
        lambda x: re.sub(pattern_removal, r'', x))
    # Удаляем из датафрейма пустые строки, cтроки состоящие из пробелов и строки не содержащие букв
    df_sentences = df_sentences.loc[(df_sentences['sentence'] != "") & ~(df_sentences['sentence'].str.isspace())
                                    & df_sentences['sentence'].str.contains(r'[A-zА-я]')]
    return df_sentences


# Функция выделения введения документа
def introduction_selector(lines):
    chapter_started = False
    chapter_lines = []
    # Выделяем необходимую главу
    for line in lines:
        if not chapter_started:
            if f"введение" in line.lower():
                chapter_started = True
        else:
            if "глава" in line.lower():
                chapter_started = False
            else:
                chapter_lines.append(line)
    return chapter_lines


# Функция выделения заключения документа
def conclusion_selector(lines):
    chapter_started = False
    chapter_lines = []
    # Выделяем необходимую главу
    for line in lines:
        if not chapter_started:
            if f"заключение" in line.lower():
                chapter_started = True
        else:
            if "список литературы" in line.lower() or "приложение" in line.lower() or "приложения" in line.lower():
                chapter_started = False
            else:
                chapter_lines.append(line)
    return chapter_lines


def summarize_sumy(text, sentence_count, Summarizer, stemming=False, stop_words=False):
    parser = PlaintextParser.from_string(text, Tokenizer("russian"))

    if stemming:
        stemmer = Stemmer("russian")
        summarizer = Summarizer(stemmer)
    else:
        summarizer = Summarizer()

    if stop_words:
        summarizer.stop_words = frozenset(nltk.corpus.stopwords.words("russian"))

    summary = summarizer(parser.document, sentence_count)
    return summary


def introduction_conclusion_extract(lines):
    chapter_started = False
    chapter_lines = []
    strings_to_find = ["заключение", "введение"]
    strings_to_stop = ["список литературы", "глава", "библиографический список"]
    string_article_ends = ["список литературы", "библиографический список"]
    article_ends = 0
    for line in lines:
        if not chapter_started:
            if any(s in line.lower() for s in strings_to_find):
                chapter_started = True
        else:
            if any(s in line.lower() for s in strings_to_stop):
                chapter_started = False
                if any(s in line.lower() for s in string_article_ends):
                    article_ends += 1
                    if (article_ends > 1):
                        break
            else:
                chapter_lines.append(line)

    return chapter_lines


def dock_processing(url_dock_address):

    with open('models/task_vectorizer.pkl', 'rb') as file:
        task_vectorizer = cloudpickle.load(file)

    with open('models/task_classifier.pkl', 'rb') as file:
        task_classifier = cloudpickle.load(file)

    # Извлечение ФИО:
    # Загрузка .word документа и преобразование в .txt
    text = docx2txt.process(f'{url_dock_address}')
    # Извлечение титульного листа
    title_page_text = get_title_page_text(text)
    # Предобработка текста титульного листа
    preprocessed_title_page_text = title_page_preprocessing(title_page_text)
    # Получение именованных сущностей(ФИО) из титульного листа
    students_names_list, leaders_names_list = get_group_names(preprocessed_title_page_text)

    # Извлечение задач:
    # Предобработаем текст
    preprocessed_text = text_preprocessing(text)
    # Разделим текст на строки
    preprocessed_text_lines = line_splitter(preprocessed_text)
    # Выделим необходимую главу
    tasks_chapter_number = 4
    tasks_chapter_lines = chapter_selector(preprocessed_text_lines, tasks_chapter_number)
    # Произведем токенизацию на предложения
    tasks_chapter_sentences = lines_to_sentences_tokenization(tasks_chapter_lines)
    # Векторизируем
    X = task_vectorizer.transform(tasks_chapter_sentences)
    # Классифицируем предложения:
    y = task_classifier.predict(X)
    # Создадим таблицу с классами предложений
    df_classified_sentences = df_sentences_preprocessing(
        pd.DataFrame({"sentence": tasks_chapter_sentences, "class": y}))
    # Сохраним задачи в список
    tasks_list = []
    for i, row in df_classified_sentences.iterrows():
        if row['class'] == 1:
            tasks_list.append(row['sentence'])

    # Извлечение ключевых слов:
    # Выделим необходимые главы
    introduction_conclusion_lines = introduction_conclusion_extract(preprocessed_text_lines)
    # Произведем токенизацию на предложения
    sentences_for_keywords = lines_to_sentences_tokenization(introduction_conclusion_lines)
    # Преобразуем строки в цельный текст
    text_for_keywords = ' '.join(sentences_for_keywords)
    # Экстрактору Yake модели зададим параметры:
    # Русскоязычная модель, фраза размерностью 2, мера схожести слов не более 50%, извелчение в количестве 10 фраз
    yakeModel = KeywordExtractor(lan="ru", n=2, dedupLim=0.5, top=10)
    # Извлечем ключевые слова
    yake_keywords = yakeModel.extract_keywords(text_for_keywords)
    keyword_list = []
    for score, keyword in yake_keywords:
        keyword_list.append(score)

    # Извлечение технологий из задач:
    technology_list = []
    technology_extractor = spacy.load("en_core_web_sm")  # загрузка модели языка
    for task in tasks_list:
        tasks_technology_list = []
        doc = technology_extractor(task)
        for entity in doc.ents:
            if entity.label_ == "PRODUCT":  # фильтрация сущностей, относящихся к продуктам/технологиям
                tasks_technology_list.append(entity.text)
        technology_list.append(tasks_technology_list)

    # Извлечение аннотации:
    introduction_conclusion_lines = [s for s in introduction_conclusion_lines if
                                     not re.search(".*изуч|осво|навык|умен|опыт|защит|квалиф|компетен.*", s.lower())]
    introduction_conclusion_text = "\n".join(introduction_conclusion_lines)
    sentence_count = 5
    sumy_summarizer = TextRankSummarizer
    summary = summarize_sumy(introduction_conclusion_text, sentence_count, sumy_summarizer, True, True)
    annotation_sentences_list = []
    for sentence in summary:
        annotation_sentences_list.append(str(sentence))

    # Упаковка списков в json
    data_to_pack = {
        "students": students_names_list,
        "leaders": leaders_names_list,
        "tasks": tasks_list,
        "technologies": technology_list,
        "keywords": keyword_list,
        "annotation_sentences": annotation_sentences_list
    }
    # Сериализуем словарь в формат JSON с декодированием символов
    packed_data_json = json.dumps(data_to_pack, ensure_ascii=False)
    return packed_data_json
