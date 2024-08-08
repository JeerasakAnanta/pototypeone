import os
import sys

# Import classes from modules
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create the language model
llm1 = ChatOpenAI(model="gpt-3.5-turbo")

# load documents from pdf 
def load_documents():
    # Define the path to your local PDF file
    local_path = "./data/ข้อมูลยา 50 ชนิด.pdf"

    # Load and split the PDF file
    if local_path:
        loader = PyMuPDFLoader(file_path=local_path)
        data = loader.load()
    else:
        print("Upload a PDF file")
        sys.exit()
        
    return data
        
def  split_documents(data): 
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    
    # This part is used for embedding the docs and storing them into Vector DB and initializing the retriever.
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    
    return  docsearch
    
def llm_roleplay():
    custom_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI thai language model assistant.
    You are an expert at answering questions about medicine.
    Answer the question based ONLY on the following context.
    
    {context}
    Original question: {question}""",
)
def main():
    
        #   load documents from pdf function 
    data = load_documents()
    docsearch = split_documents(data)
    custom_template = llm_roleplay()
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
    llm=llm1,
    retriever=docsearch.as_retriever(),
    combine_docs_chain_kwargs={"prompt": custom_template}

    )
    
    chat_history = []
    query = None  # Initialize query to avoid potential reference error

    while True:
        if not query:
            query = input("User: ")
        if query in ['quit', 'q', 'exit']:
            break
        result = chain.invoke({"question": query, "chat_history": chat_history})
        print("Chatbot:", result['answer'])

        chat_history.append((query, result['answer']))
        query = None
    
    print("Goodbye!")
if __name__ == '__main__':
    main()