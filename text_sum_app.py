import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader,UnstructuredURLLoader


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    
generic_url=st.text_input("URL",label_visibility="collapsed")

#model 
llm = ChatGroq(api_key=groq_api_key,model="gemma2-9b-it")

#promptTemplate for Summarization
chunks_prompt="""
Provide a detailed summary of the following content:
Content:{text}
"""
map_prompt_template=PromptTemplate(template=chunks_prompt,input_variables=["text"])


final_prompt="""
provide the final summary of the entire content from these important points
content:{text}"""
final_template = PromptTemplate(template=final_prompt,input_variables=["text"])




if st.button("Summarize the content from Youtube or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    
    else:
        try:
             with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(docs)
                chain=load_summarize_chain(llm,chain_type="map_reduce",
                                           map_prompt=map_prompt_template,
                                           combine_prompt=final_template,
                                           verbose=True)
                output_summary=chain.run(chunks)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")