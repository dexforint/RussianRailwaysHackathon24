from enum import Enum

model = None


class LLMModel(str, Enum):
    VIKHR = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"


def get_llm_function(model_name: LLMModel = LLMModel.VIKHR):
    """Функция для получения функции ответа от LLM"""
    global model
