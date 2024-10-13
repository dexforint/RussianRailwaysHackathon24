import faiss
import numpy as np
import pickle
import os


# if os.path.exists("./data/summary_index.faiss"):
#     summary_index = faiss.read_index("./data/summary_index.faiss")
# else:
#     summary_index = None

if os.path.exists("./data/chunk_index.faiss"):
    chunk_index = faiss.read_index("./data/chunk_index.faiss")
else:
    chunk_index = None

if os.path.exists("./data/questions_index.faiss"):
    questions_index = faiss.read_index("./data/questions_index.faiss")
else:
    questions_index = None

if os.path.exists("./data/keywords_index.faiss"):
    keywords_index = faiss.read_index("./data/keywords_index.faiss")
else:
    keywords_index = None


# Конфигурация для работы с FAISS на GPU
def get_faiss_index(dimension, use_gpu=True):
    # Создаем индекс для векторов размерности 'dimension'
    index = faiss.IndexFlatIP(dimension)

    if use_gpu:
        # Переносим индекс на GPU
        res = faiss.StandardGpuResources()  # создаем GPU-ресурсы
        index = faiss.index_cpu_to_gpu(
            res, 0, index
        )  # индекс на GPU (0 - ID устройства)

    return index


# Функция для добавления данных в индекс
def add_to_index(index, vectors):
    index.add(vectors)  # Добавляем векторы в индекс


# Сохранение и загрузка индекса
def save_index(index, file_path):
    index_cpu = faiss.index_gpu_to_cpu(
        index
    )  # Переносим индекс с GPU на CPU перед сохранением
    faiss.write_index(index_cpu, file_path)  # Сохраняем индекс на диск


def load_index(file_path, use_gpu=True):
    index_cpu = faiss.read_index(file_path)  # Загружаем индекс с диска
    if use_gpu:
        # Переносим индекс на GPU
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        return index_gpu
    return index_cpu


indexes = {
    "summary": (
        load_index("./data/summary_index.faiss"),
        None,
    ),
    "keywords": (
        load_index("./data/keyword_index.faiss"),
        pickle.load(open("./data/keyword_id2chunk_id.pkl", "rb")),
    ),
    "passage_questions": (
        load_index("./data/passage_question_index.faiss"),
        pickle.load(open("./data/question_id2chunk_id.pkl", "rb")),
    ),
    "query_questions": (
        load_index("./data/query_question_index.faiss"),
        pickle.load(open("./data/question_id2chunk_id.pkl", "rb")),
    ),
}


# Функция для поиска ближайших векторов
def search_in_index(index_name, query_vectors, k=32):
    index, id2chunk_id = indexes[index_name]
    distances, indices = index.search(query_vectors, k)  # Поиск ближайших k векторов

    chunk_id2score = {}
    for dist, idx in zip(distances, indices):
        chunk_id = id2chunk_id[idx]
        chunk_id2score[chunk_id] = 1 - (dist + 1) / 2

    return chunk_id2score


def main():
    # Параметры
    dimension = 128  # Размерность векторов
    num_vectors = 1000  # Количество векторов для добавления
    k = 5  # Количество ближайших соседей для поиска
    db_file = "faiss_index.bin"  # Имя файла для сохранения базы данных

    # Генерируем случайные векторы
    np.random.seed(42)
    vectors = np.random.random((num_vectors, dimension)).astype("float32")

    # Создаем индекс
    index = get_faiss_index(dimension)

    # Добавляем векторы в индекс
    add_to_index(index, vectors)
    print(f"Добавлено {num_vectors} векторов в индекс.")

    # Сохраняем индекс на диск
    save_index(index, db_file)
    print(f"Индекс сохранен в файл: {db_file}")

    # Загружаем индекс с диска
    loaded_index = load_index(db_file)
    print(f"Индекс загружен из файла: {db_file}")

    # Генерируем запрос для поиска (например, возьмем первые 5 векторов из исходных данных)
    query_vectors = vectors[:5]

    # Выполняем поиск по загруженному индексу
    distances, indices = search_in_index(loaded_index, query_vectors, k)

    print("Результаты поиска:")
    for i, (d, idx) in enumerate(zip(distances, indices)):
        print(f"Запрос {i}:")
        for dist, ind in zip(d, idx):
            print(f"  Вектор {ind} с расстоянием {dist}")

    # Пример добавления новых векторов в индекс
    new_vectors = np.random.random((10, dimension)).astype("float32")
    add_to_index(loaded_index, new_vectors)
    print(f"Добавлено 10 новых векторов в индекс.")


if __name__ == "__main__":
    main()
