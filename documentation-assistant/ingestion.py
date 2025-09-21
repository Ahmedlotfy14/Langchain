import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import pinecone
from consts import INDEX_NAME
import pinecone

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT'],
)

def ingest() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    raw_doc = loader.load()
    print(f"loaded {len(raw_doc)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,chunk_overlap=100,separators=["\n\n","\n"," ",""]
    )
    print(f"splitting {len(text_splitter)} documents")
    doc = text_splitter.split_documents(documents=raw_doc)
    embeddings = OpenAIEmbeddings()

    for document in doc:
        old_path = document.metadata["source"]
        new_path = old_path.Replace("langchain-docs", "https:/")
        document.metadata.update({"source": new_path})

    print(f"saving {len(doc)} documents to pinecone")
    embeddings = OpenAIEmbeddings()
    pinecone.from_documents(documents=doc, embeddings=embeddings, inedx_name=INDEX_NAME)
    print("***** saved to pinecone store")

if __name__ == "__main__":
    ingest()


