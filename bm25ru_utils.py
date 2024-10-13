# pip install rank-bm25 pymorphy2 nltk

import sqlite3
import nltk
import pymorphy2
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json

nltk.download("punkt")

# Инициализация морфологического анализатора для русского языка
morph = pymorphy2.MorphAnalyzer()


# Функция для лемматизации (приведение слов к начальной форме)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    lemmas = [
        morph.parse(token)[0].normal_form for token in tokens if token.isalpha()
    ]  # Лемматизация
    return lemmas


class BM25Database:
    def __init__(self, db_path="bm25_database.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        self.corpus = []
        self.documents = []  # Список для хранения документов
        self.bm25 = None

    # Создание таблицы для хранения документов
    def create_table(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT
                )
            """
            )

    # Добавление документа в базу данных
    def add_document(self, content):
        preprocessed_content = preprocess_text(content)
        with self.conn:
            self.conn.execute(
                "INSERT INTO documents (content) VALUES (?)",
                (json.dumps(preprocessed_content),),
            )
        self.documents.append(content)
        self.corpus.append(preprocessed_content)
        self.update_bm25()

    # Обновление модели BM25 при изменении корпуса
    def update_bm25(self):
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    # Загрузка всех документов из базы данных
    def load_documents(self):
        with self.conn:
            cursor = self.conn.execute("SELECT content FROM documents")
            rows = cursor.fetchall()
            self.corpus = [json.loads(row[0]) for row in rows]
            self.documents = [" ".join(doc) for doc in self.corpus]
            self.update_bm25()

    # Поиск документов по запросу
    def search(self, query, top_n=5):
        if not self.bm25:
            return []

        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        top_n_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]
        results = [(self.documents[i], scores[i]) for i in top_n_indices]
        return results

    # Сохранение базы данных (закрытие соединения)
    def save(self):
        self.conn.commit()
        self.conn.close()

    # Загрузка существующей базы данных
    @staticmethod
    def load(db_path="bm25_database.db"):
        db = BM25Database(db_path)
        db.load_documents()
        return db


# Создание новой базы данных
db = BM25Database()

# Добавление документов
db.add_document("Это первый документ на русском языке.")
db.add_document("Второй документ посвящен другой теме.")
db.add_document("Этот текст содержит информацию о BM25 и поиске по тексту.")

# Поиск по базе данных
query = "поиск текста"
results = db.search(query)
for doc, score in results:
    print(f"Документ: {doc}, Релевантность: {score}")

# Сохранение базы данных
db.save()

# Загрузка базы данных и поиск
db = BM25Database.load()
results = db.search("документ")
for doc, score in results:
    print(f"Документ: {doc}, Релевантность: {score}")
