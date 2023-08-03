from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os
load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader (file_path="data.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents (documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search (query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
template= """
You are an assistant who helps customers understand our company. I will provide a question from a customer
and you will give them the best answer based on some of our data.

Below is a question I received from the customer:
{message}

Here is a list of related data about our company:
{company_data}
Please write the best response for the customer.

IMPORTANT NOTE: If the question is not related to our company, don't give direct answer but let the customer know that question is out-of-scope for this conversation and then get customer's attention back to the context of our company.
"""

prompt = PromptTemplate(
    input_variables=["message", "company_data"],
    template=template
)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    company_data = retrieve_info(message)
    response = chain.run(message=message, company_data=company_data)
    return response

# 5. test
# message = """
# why should I choose your service?
# """

# response = generate_response(message)
# print(response)
