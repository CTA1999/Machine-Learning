

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.title("Student Advisor")

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()

# Note that you can use PromptTemplate to instruct model to use specific rules if you don't want to use default
prompt_template = PromptTemplate(template = """
You are a helpful assistant.

Answer the user’s question using ONLY the context provided below.

If the answer is not contained in the context, respond with EXACTLY:
"I’m sorry, I am only authorized to talk about the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
 """,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4} ), # Number of chunks to retrieve
   chain_type_kwargs={"prompt": prompt_template} # We use default behavior here
)

question = st.text_input(f"Ask a question about your documents:")

if question:
    with st.spinner("Searching documents..."):
        response = qa_chain.run(question)
        st.success("Answer:")
        st.write(response)

        with st.expander("View source passages"):
            docs = vectorstore.similarity_search(question, k=3) # k is the number of sources you'd like to see listed.
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                st.text(doc.page_content[:500] + "...")
                st.divider()