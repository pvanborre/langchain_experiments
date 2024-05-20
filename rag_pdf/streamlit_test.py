import streamlit as st
from PyPDF2 import PdfReader
from summarizer import Summarizer  

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    summarizer = Summarizer()  
    summary = summarizer(text)
    return summary

def main():
    st.title("PDF Summarizer")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner('Loading and summarizing...'):
            
            text = load_pdf(uploaded_file)
            
            summary = summarize_text(text)
            
            st.subheader("Summary")
            st.write(summary)
        st.success('Done!')

if __name__ == "__main__":
    main()
