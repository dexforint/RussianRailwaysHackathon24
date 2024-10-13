import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Загрузка токенизатора nltk
nltk.download("punkt")


class BM25Database:
    def __init__(self, documents=None):
        """
        Инициализация базы данных BM25. Если передан список документов, то база создается на его основе.
        """
        self.documents = documents if documents else []
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add_document(self, document):
        """
        Добавляет документ в базу данных и обновляет индекс BM25.
        """
        self.documents.append(document)
        tokenized_document = word_tokenize(document.lower())
        self.tokenized_corpus.append(tokenized_document)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_n=5):
        """
        Выполняет поиск по базе данных с использованием BM25.

        :param query: Запрос для поиска.
        :param top_n: Количество топовых результатов для возврата.
        :return: Список документов, отсортированных по релевантности.
        """
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]
        results = [(self.documents[i], scores[i]) for i in top_n_indices]
        return results

    def save_database(self, filename):
        """
        Сохраняет базу данных в файл с использованием pickle.

        :param filename: Имя файла для сохранения базы данных.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.documents, f)

    def load_database(self, filename):
        """
        Загружает базу данных из файла с использованием pickle.

        :param filename: Имя файла для загрузки базы данных.
        """
        with open(filename, "rb") as f:
            self.documents = pickle.load(f)
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)


# Пример использования
if __name__ == "__main__":
    # Создание новой базы данных
    bm25_db = BM25Database()

    # Добавление документов
    bm25_db.add_document("This is a sample document about machine learning.")
    bm25_db.add_document(
        "Another document related to data science and machine learning."
    )
    bm25_db.add_document("This is a completely different topic about cooking.")

    # Поиск по базе данных
    results = bm25_db.search("machine learning", top_n=3)
    for doc, score in results:
        print(f"Document: {doc}, Score: {score}")

    # Сохранение базы данных
    bm25_db.save_database("bm25_database.pkl")

    # Загрузка базы данных
    bm25_db.load_database("bm25_database.pkl")

    # Повторный поиск после загрузки
    results = bm25_db.search("data science", top_n=3)
    for doc, score in results:
        print(f"Document: {doc}, Score: {score}")
