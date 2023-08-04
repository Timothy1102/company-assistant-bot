from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os
load_dotenv()

# 1. Vectorise company csv data
loader = CSVLoader(file_path="data.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
template= """
You are an assistant who helps me understand your company. I will provide a question
and you will give me the best answer based on some data about your company.

Below is my question:
{message}

Here is a list of related data about your company:
{company_data}
Please write the best response.

IMPORTANT NOTE: If the question is not related to your company, don't give direct answer but let me know that question is out-of-scope for this conversation and then get my attention back to the context of your company.
"""

prompt = PromptTemplate(
    input_variables=["message", "company_data"],
    template=template
)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Get augmented generation
def generate_response(message):
    company_data = retrieve_info(message)
    response = chain.run(message=message, company_data=company_data)
    return response

# 5. Test
# message = """
# why should I choose your service?
# """

# response = generate_response(message)
# print(response)
