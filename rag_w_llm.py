import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

class KnowledgeBase:
    def __init__(self, data_path='data', keep_prev=False, chroma_path='chroma'):
        self.data_path = data_path
        self.chroma_path = chroma_path
        if not keep_prev:
            print("Clearing Knowledge Database")
            self.clear_database()

        # Create or update the database.
        self.update_database()
        self.prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """)

        self.model = None

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

    def update_database(self):
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks)

    def load_documents(self):
        document_loader = PyPDFDirectoryLoader(self.data_path)
        return document_loader.load()

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def add_to_chroma(self, chunks: list[Document]):
        # Load the existing database.
        db = Chroma(
            persist_directory=self.chroma_path, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # only retrieve ID
        existing_ids = set(existing_items["ids"])

        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        new_chunk_ids = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
                new_chunk_ids.append(chunk.metadata["id"])
        if new_chunks:
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print(f"Add {len(new_chunks)} new documents to DB.")
        else:
            print("No new documents to add")

    def calculate_chunk_ids(self, chunks):
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        last_chunk_index = -1

        for chunk in chunks:
            source = chunk.metadata['source']
            page = chunk.metadata['page']
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index = last_chunk_index + 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            chunk.metadata["id"] = chunk_id

            last_chunk_index = current_chunk_index
            last_page_id = current_page_id
        return chunks

    def query_rag(self, query_text: str):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=get_embedding_function())

        search_results = db.similarity_search_with_score(query_text, k=3)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in search_results])
        prompt = self.prompt_template.format(context=context_text, question=query_text)

        if self.model is None:
            self.model = Ollama(model="mistral")
        response_text = self.model.invoke(prompt)

        sources = [doc.metadata['id'] for doc, _ in search_results]
        print(f"Response: {response_text}\nSources: {sources}")
        return response_text


if __name__ == "__main__":
    my_db = KnowledgeBase(keep_prev=True)

    my_db.query_rag("When is wedding season in ACNH? ")
