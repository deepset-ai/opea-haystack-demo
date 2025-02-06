import streamlit as st
from dotenv import load_dotenv

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def set_initial_state():
    load_dotenv()
    set_state_if_absent("question", "")
    set_state_if_absent("results", None)
    set_state_if_absent("documents", None)

def reset_results(*args):
    st.session_state.results = None
    st.session_state.documents = None
    st.session_state.retrieval_type = None
    
def sidebar():
    with st.sidebar:
        
        st.image('logo/haystack-logo.png')
        st.image('logo/opea-logo.svg')
        st.markdown("# Demopalooza")
        
        st.link_button("Go to Code", "https://github.com/deepset-ai/opea-haystack-demo/tree/main")
        
        st.markdown("## 📚 Useful Resources\n"
                    "* [Get Started](https://haystack.deepset.ai/overview/quick-start)\n"
                    "* [Docs](https://docs.haystack.deepset.ai/docs/intro)\n"
                    "* [Tutorials](https://haystack.deepset.ai/tutorials)\n"
                    "* [Integrations](https://haystack.deepset.ai/integrations)\n"
        )
   
def haystack_tab():
    st.markdown('''## Haystack''')
    st.markdown("""
    * Fully open-source framework built in Python 
         for custom AI applications by deepset
    * Provides tools that developers need to 
         build and customize state-of-the-art AI systems
    * Building blocks: Components & Pipelines
               """)
    
    st.image('images/pipeline.png')
   
def pipeline_tab():
    st.markdown("""
                    ### Dataset
                    **ARAGOG**: This dataset is based on the paper Advanced Retrieval Augmented Generation Output Grading (ARAGOG). 
                    It's a collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format.
                    * 13 PDF papers.
                    * 107 questions and answers generated with the assistance of GPT-4, and validated/corrected by humans.
                    """)
    st.markdown('''### Indexing Pipeline''') 
    st.image('images/indexing.png')
    with st.expander("Indexing Pipeline Code"): 
        st.code("""
    document_store  = InMemoryDocumentStore()
    file_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_converter = TextFileToDocument()
    pdf_converter = PyPDFToDocument() # requires 'pip install pypdf'
    markdown_converter = MarkdownToDocument()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=10)
    document_embedder = OPEADocumentEmbedder("http://localhost:6006")
    document_joiner = DocumentJoiner()
    document_writer = DocumentWriter(document_store=document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=file_router, name="file_router")
    indexing_pipeline.add_component(instance=text_converter, name="text_converter")
    indexing_pipeline.add_component(instance=pdf_converter, name="pdf_converter")
    indexing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")    
    indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    indexing_pipeline.add_component(instance=document_writer, name="document_writer")

    indexing_pipeline.connect("file_router.text/plain", "text_converter")
    indexing_pipeline.connect("file_router.application/pdf", "pdf_converter")
    indexing_pipeline.connect("file_router.text/markdown", "markdown_converter")
    indexing_pipeline.connect("markdown_converter", "document_joiner")
    indexing_pipeline.connect("text_converter", "document_joiner")
    indexing_pipeline.connect("pdf_converter", "document_joiner")
    indexing_pipeline.connect("document_joiner", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")
    
    files_path = "files"
    pdf_files = [files_path+"/"+f_name for f_name in os.listdir(files_path)]
    indexing_pipeline.run(data={"file_router":{"sources":pdf_files}})
    """, language="python")
    st.markdown('''### RAG with Keyword-based Retrieval''')
    st.image('images/keyword_rag.png')
    st.markdown("""
    ```python
        prompt='''Answer the following query based on the given documents.

    Documents:
    {% for document in documents %}
        {{document.content}}
    {% endfor %}

    Query: {{question}}

    If you have enough context to answer this question, just return your answer
    If you don't have enough context to answer, say 'N0_ANSWER'.'''  
    ```            
                """)    
    with st.expander("Keyword-based Retrieval Pipeline Code"): 
        st.code("""          
    keyword_search_rag = AsyncPipeline()
    keyword_search_rag.add_component("retriever", InMemoryBM25Retriever(_document_store, top_k=5))
    keyword_search_rag.add_component("prompt_builder", PromptBuilder(template=prompt))
    keyword_search_rag.add_component("generator", OPEAGenerator("http://localhost:9009", model_arguments={"temperature": 0.9, "top_p": 0.7, "max_tokens": 100}))
    
    keyword_search_rag.connect("retriever.documents", "prompt_builder.documents")
    keyword_search_rag.connect("prompt_builder", "generator")
    
    keyword_search_rag.run_async({
        "retriever": {"query": question},
        "prompt_builder": {"question": question}
    })
    """, language="python")
    st.markdown('''### RAG with Embedding Retrieval''')
    st.image('images/embedding_rag.png')
    with st.expander("Embedding Retrieval Pipeline Code"): 
        st.code("""
    embedding_rag = AsyncPipeline()
    embedding_rag.add_component("embedder", OPEATextEmbedder("http://localhost:6006"))
    embedding_rag.add_component("adapter", OutputAdapter(template="{{ embedding[0] }}", output_type=List[float]))
    embedding_rag.add_component("retriever", InMemoryEmbeddingRetriever(_document_store, top_k=5))
    embedding_rag.add_component("prompt_builder", PromptBuilder(template=prompt))
    embedding_rag.add_component("generator", OPEAGenerator("http://localhost:9009", model_arguments={"temperature": 0.9, "top_p": 0.7, "max_tokens": 100}))
    
    embedding_rag.connect("embedder.embedding", "adapter.embedding")
    embedding_rag.connect("adapter", "retriever.query_embedding")
    embedding_rag.connect("retriever.documents", "prompt_builder.documents")
    embedding_rag.connect("prompt_builder", "generator")
    
    embedding_rag.run_async({
        "embedder": {"text": question},
        "prompt_builder": {"question": question}
    })
    """, language="python")
        
    st.markdown('''### RAG with Hybrid Retrieval (BM25 + Embedding)''')
    st.image('images/hybrid_rag.png')
    with st.expander("Hybrid Retrieval Pipeline Code"): 
        st.code("""
    hybrid_rag = AsyncPipeline()
    hybrid_rag.add_component("embedder", OPEATextEmbedder("http://localhost:6006"))
    hybrid_rag.add_component("bm25_retriever", InMemoryBM25Retriever(_document_store, top_k=5))
    hybrid_rag.add_component("retriever", InMemoryEmbeddingRetriever(_document_store, top_k=5))
    hybrid_rag.add_component("adapter", OutputAdapter(template="{{ embedding[0] }}", output_type=List[float]))
    hybrid_rag.add_component("joiner", DocumentJoiner())
    hybrid_rag.add_component("prompt_builder", PromptBuilder(template=prompt))
    hybrid_rag.add_component("generator", OPEAGenerator("http://localhost:9009", model_arguments={"temperature": 0.9, "top_p": 0.7, "max_tokens": 100}))
    
    hybrid_rag.connect("embedder.embedding", "adapter.embedding")
    hybrid_rag.connect("adapter", "retriever.query_embedding")
    hybrid_rag.connect("retriever", "joiner")
    hybrid_rag.connect("bm25_retriever", "joiner")
    hybrid_rag.connect("joiner", "prompt_builder.documents")
    hybrid_rag.connect("prompt_builder", "generator")
        
    hybrid_rag.run_async({
        "embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "prompt_builder": {"question": question}
    })
    """, language="python")     

def next_steps_tab():
    st.markdown("""## Next with OPEA x Haystack""")
    st.markdown("""
    * RAG ✅
    * Web RAG
    * Converting, preprocessing, embedding, indexing ✅
    * Advanced Retrieval (Hybrid ✅, Sentence Window Retrieval, HyDE)
    * Conversational Systems, Chat
    * Agent (Tool Calling, Self Reflection, ReAct, ... )
                """)
    st.image('images/self-reflecting.png', caption="Self-Reflecting Agent")
    st.image('images/web-enhanced.png', caption="Web-Enhanced RAG Pipeline")