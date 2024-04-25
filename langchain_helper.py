from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature=0.1)

loader = CSVLoader(file_path='Q&A.csv', source_column='prompt')
data = loader.load()

instructor_embeddings = HuggingFaceInstructEmbeddings()

vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
