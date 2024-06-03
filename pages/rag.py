import streamlit as st
from langchain_core.prompts import PromptTemplate
from st_pages import add_page_title

from src.rag import RagModel
from src.utils import llm, rag

st.set_page_config(page_title='🦜 Nexa')

PROMPT_TEMPLATE = """<|im_start|>\nuser\nОтветь на вопрос базируясь только на этом контексте:

{context}

---
Ответь на вопрос, используя только контекст: {question}<|im_end|>
"""


# rag = RagModel()


def generate_response(dialogue: str):
    return llm(dialogue, max_tokens=2000, repeat_penalty=1.0)


add_page_title()
st.text("Система расширенного поиска")

if question := st.chat_input():
    with st.spinner("Подождите ..."):
        output = rag.generate(query_text=question)
        if len(output) == 0 or output[0][1] < 0.7:
            response = "Нет фрагментов текста, на которые можно опираться для ответа."
        else:

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in output])

            prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

            prompt = prompt_template.format(context=context_text, question=question)

            response = generate_response(dialogue=prompt)['choices'][0]['text'].replace('assistant', '').replace(
                'ssistant', '')

        st.write(response)
