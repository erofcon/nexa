import random

from llama_cpp import Llama
import streamlit as st

from src.rag import RagModel

llm: Llama | None = None
rag: RagModel | None = None


@st.cache_resource
def set_model(model_path="model_path/model-q2_K.gguf"):
    global llm
    llm = Llama(model_path, verbose=True, n_ctx=8192, n_gpu_layers=32, max_tokens=1000)


@st.cache_resource
def set_rag():
    global rag
    rag = RagModel()
    rag.create()

    # llm = LlamaCpp(model_path=model_path, verbose=True, n_ctx=8192, n_gpu_layers=32, max_tokens=1000)

# from src.model import LocalLlama
#
# SAMPLE_QUESTIONS = [
#     "Напиши сатиру на успешный успех",
#     "Какая энергия связывает кварки в нуклонах?",
#     "Напиши содержание «Войны и мира» в 200 знаках",
#     "О чем будет следующий роман Пелевина?",
#     "Проведи антонимы (горячий-холодный)",
#     "Напиши хорошее резюме для человека без опыта работы, с высшим образованием",
#     "Исправь мой корявый текст в нормальный",
#     "Придумай историю про Чапаева и Чака Нориса?",
#     "Составь список упражнений для 3 пункта",
#     "Пошаговая система для погружения в любую новую тему",
#     "Новые идей, когда не знаешь, с чего начать",
#     "Напиши коммерческое предложение для оптовых закупок систем видеонаблюдение",
#     "Напиши рекомендации для общения сотрудников технической поддержки с клиентами",
#     "Напиши пожалуйста шаблон для рекламы курсов по математике",
#     "Сделай коммерческое предложение инвестиция в спорт",
#     "Придумай 5 гиперонимов слову яблоко",
# ]
#
# PLACEHOLDER = random.choice(SAMPLE_QUESTIONS)
#
#
# def generate_response(llm: LocalLlama, dialog: str):
#     return llm.call()(dialog, max_tokens=1000, repeat_penalty=1.0)
#     # return llm(dialog, max_tokens=1000, repeat_penalty=1.0)
