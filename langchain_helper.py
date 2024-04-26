from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature=0.1)

loader = CSVLoader(file_path='Q&A.csv', source_column='prompt')
data = loader.load()

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large")

vectordb = FAISS.from_documents(
    documents=docs, embedding=instructor_embeddings)

retriever = vectordb.as_retriever
rdoc = retriever.get_relevant_documents("for how long is this program")

RetrievalQA(llm=llm, chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT})
