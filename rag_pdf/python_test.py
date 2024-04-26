from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import MathpixPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain



loader = PyPDFium2Loader("allcott.pdf")
docs = loader.load()
print(docs)


embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

mini_documents = documents[0:10]

vector = FAISS.from_documents(mini_documents, embeddings)


llm = Ollama(model="llama2")



prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "could you give me more details about the survey authors conducted?"})
print(response["answer"])