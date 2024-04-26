from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature=0.1)


instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large")


def create_vector_db():
    loader = CSVLoader(file_path='Q&A.csv', source_column='prompt')
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)


retriever = vectordb.as_retriever
rdoc = retriever.get_relevant_documents("for how long is this program")

prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

RetrievalQA(llm=llm, chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT})
