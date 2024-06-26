import pathway as pw
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature=0.1)


instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large")

vectordb_file_path = "faiss_index"


def create_vector_db():

    data_stream = pw.io.fs.read(
        path="Q&A.csv", mode="streaming", format="text", autocommit_duration_ms=50
    )

    parser = pw.xpacks.llm.parsers.ParseUnstructured()
    documents = data_stream.select(texts=parser(pw.this.data))
    documents = documents.flatten(pw.this.texts)

    embedded_data = pw.call(
        "embedder", context=documents, data_to_embed=pw.this.texts, model=instructor_embeddings
    )

    vectordb = FAISS.from_documents(documents=embedded_data)

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_vector_db()

print(chain("how long is the program"))