import os
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# TODO: refactor


class RagModel:

    def __init__(self):
        self.db = None

    def _get_embeddings(self):
        # embedding_function = SentenceTransformerEmbeddingFunction()
        model_kwargs = {'device': 'cuda'}
        embeddings_hf = HuggingFaceEmbeddings(
            model_name='intfloat/multilingual-e5-large',
            model_kwargs=model_kwargs
        )
        return embeddings_hf
        # return embedding_function

    def _load_documents(self) -> list[Document]:
        loader = DirectoryLoader('documents', glob="*.txt")
        documents = loader.load()
        return documents

    def _split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)

        # TODO: add logging

        print(f'Smashed {len(documents)} documents on {len(chunks)} chunks')

        return chunks

    def _save_to_chroma(self, chunks: list[Document]):
        if os.path.exists('chroma_db'):
            shutil.rmtree('chroma_db')

        db = Chroma.from_documents(
            chunks, self._get_embeddings(), persist_directory='chroma_db'
        )
        db.persist()

        # TODO: add logging
        print(f"Saved {len(chunks)} chunks to {'chroma_db'}.")

    def prepare(self):
        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._save_to_chroma(chunks)

    def create(self):
        self.db = Chroma(persist_directory='chroma_db', embedding_function=self._get_embeddings())

    def generate(self, query_text: str):
        return self.db.similarity_search_with_relevance_scores(query_text)


if __name__ == '__main__':
    RagModel().prepare()
