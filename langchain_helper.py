from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import os

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature= 0.1)

result = llm(input('Type here '))

print(result)