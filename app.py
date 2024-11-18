# %%
# Update these imports at the top of your file
from langchain_community.vectorstores import Chroma  # For creating the database for PDF embeddings
from langchain_community.document_loaders import PyPDFLoader  # To load PDF documents
from langchain_community.embeddings import HuggingFaceEmbeddings  # For text embeddings
from langchain_openai import ChatOpenAI  # To initialize the LLM chat model

# The rest of your imports remain the same
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import css,expander_css, user_template, bot_template
import os

#%%
load_dotenv()


#%%
def handle_user_input(query):
    #get the answer from the conversation chain
    context = st.session_state.conversation_chain({"question": query, 
                                                   "chat_history": st.session_state.history}, 
                                                  return_only_outputs=True)
    
    
    
    # Append the query and answer to chat history
    st.session_state.chat_history.append((query, context["answer"]))
    
    # Extract page number from the first source document
    st.session_state.N = context['source_documents'][0].metadata.get('page', 0)
    
    
    for _, message in enumerate(st.session_state.chat_history):
        st.session_state.expander1.write(user_template.replace("{{MSG}}", message[0]), 
                                          unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", message[1]), 
                                          unsafe_allow_html=True)
        
    

# Define the process_file function
#provided model for creating embeddings
def process_file(pdf_file):
    
    #used PyPDFLoader to properly process the PDF into documents that have the required page_content attribute
    with NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        pdf_document = loader.load()
    # Creating the embeddings object using HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Example HuggingFace model
        model_kwargs={"device": "cpu"},  # or "cuda" if you want to use GPU
        encode_kwargs={"normalize_embeddings": True}  # Normalize the embeddings
    )
    
     # Define the persist directory path
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    # Step 2: Create the vector store using Chroma
    # Assume the document is in PDF format and needs to be processed
    vector_store = Chroma.from_documents(
        pdf_document, 
        embedding = embedding_model,
        persist_directory=persist_dir
    )
    vector_store.persist()
    
    # Step 3: Create the Conversational Retrieval Chain
    # Initialize the ChatOpenAI model for LLM
    llm = ChatOpenAI(temperature=0.7)  # You can adjust the temperature for more creative or focused responses

    # Create the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # Return top 5 nearest neighbors
        return_source_documents=True  # Return the source document as part of the response
    )
    
    return conversation_chain


def main():
    #load the API keys from .env file
    load_dotenv(override=True)

    
    # Set the page layout and title
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon=":books:",
        layout="wide")
    # Render the CSS in the Streamlit app
    st.write(css,unsafe_allow_html=True)

    # Set the title of the app
    st.title("RAG Chatbot")

    # Initialize session state variables if they are not present
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Changed from 'history' to 'chat_history'
    if "history" not in st.session_state:
        st.session_state.history = []  # Keep this if you need it elsewhere
    if "N" not in st.session_state:
        st.session_state.N = 0
        
    
   # Create two columns
    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    
    
    #COLUMN 1
    # Add a header to the first column
    st.session_state.col1.header("Interactive Reader :books:")
    #creating a subheader 
    st.session_state.col1.subheader("Your PDF files")
    #uploading the PDF file
    st.session_state.pdf_file = st.session_state.col1.file_uploader("Upload your \
                                                                        PDF file and click on Submit",type="pdf")
    #a button to process the file

    if st.session_state.col1.button("Submit", key = "a"):
        with st.spinner("Processing..."):
            if st.session_state.pdf_file is not None:
                #saving the conversation chain to the session state
                st.session_state.conversation_chain = process_file(st.session_state.pdf_file)
                #message to the user
                st.success("File processed successfully!")
                st.session_state.col1.markdown("Done Processing. You may now ask questions about the uploaded PDF file.")

                

    

    # Provide a text box for user input and save in a variable
    user_question = st.session_state.col1.text_input("Ask a question on the \
                                                     contents of the uploaded PDF:")
    # Inject custom CSS for the expander
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    # Add a styled expander
    st.session_state.col1.markdown(expander_css,unsafe_allow_html=True)
    
    if user_question:
        handle_user_input(user_question)
        with NamedTemporaryFile(delete=False,suffix=".pdf") as temp1:
            temp1.write(st.session_state.pdf_file.getvalue())
            temp1.seek(0)
            reader = PdfReader(temp1)
            #creating a PDF writer object
            pdf_writer = PdfWriter()
            #getting the start and end pages to extract from the PDF
            start = max(st.session_state.N - 2,0)
            end = min(st.session_state.N+2, len(reader.pages)-1 )
            while start <= end:
                #adding the pages to the PDF writer object
                pdf_writer.add_page(reader.pages[start])
                start += 1
            with NamedTemporaryFile(delete=False,suffix=".pdf") as temp2:
                pdf_writer.write(temp2.name)
                with open(temp2.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}"\
                        width="700" height="1000" type="application/pdf frameborder="0"></iframe>'
                    #displaying the PDF in the second column
                    #COLUMN 2
                    # # In the second column, you can add the PDF viewer or any other component you want
                    st.session_state.col2.header("PDF Viewer")
                    st.session_state.col2.write("PDF will be displayed here.") 
                    st.session_state.col2.markdown(pdf_display,unsafe_allow_html=True)
    #     # Submit button to send the question
    #     if st.button("Submit"):
    #         # Here you would add logic to handle the question and get the answer
    #         answer = "Sample answer based on your logic"  # Replace with your answer logic
    #         st.session_state.history.append((question, answer))  # Save question and answer to history






if __name__ == "__main__":
    main()



