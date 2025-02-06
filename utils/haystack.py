import streamlit as st

import sys
import os
sys.path.insert(0, "./GenAIComps/comps/integrations/haystack/src")

from opea_haystack.embedders.tei import OPEATextEmbedder, OPEADocumentEmbedder
from opea_haystack.generators import OPEAGenerator

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import TextFileToDocument, PyPDFToDocument, MarkdownToDocument, OutputAdapter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.evaluators import SASEvaluator
from haystack_experimental.core import AsyncPipeline
from typing import List
import asyncio

@st.cache_resource(show_spinner=False)
def start_document_store():
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    
    document_store  = InMemoryDocumentStore()
    file_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_converter = TextFileToDocument()
    pdf_converter = PyPDFToDocument() # requires 'pip install pypdf'
    markdown_converter = MarkdownToDocument()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=10) # requires 'pip install nltk'
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
    return document_store

# cached to make index and models load only at start
@st.cache_resource(show_spinner=False)
def start_haystack_pipeline(_document_store):
    prompt = """
    Answer the following query based on the given documents.

    Documents:
    {% for document in documents %}
        {{document.content}}
    {% endfor %}

    Query: {{question}}

    If you have enough context to answer this question, just return your answer
    If you don't have enough context to answer, say 'N0_ANSWER'.
    """
    
    keyword_search_rag = AsyncPipeline()
    keyword_search_rag.add_component("retriever", InMemoryBM25Retriever(_document_store, top_k=5))
    keyword_search_rag.add_component("prompt_builder", PromptBuilder(template=prompt))
    keyword_search_rag.add_component("generator", OPEAGenerator("http://localhost:9009", model_arguments={"temperature": 0.9, "top_p": 0.7, "max_tokens": 100}))
    
    keyword_search_rag.connect("retriever.documents", "prompt_builder.documents")
    keyword_search_rag.connect("prompt_builder", "generator")
    ######### Embedding Retrieval ##########
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
    
    return keyword_search_rag, embedding_rag, hybrid_rag

@st.cache_data(show_spinner=False)
def query_with_method(_pipelines, question):
    print(f"Pipeline will run with query: {question}")
    
    async def run_pipe(pipeline, data, retrieval_type):
        if retrieval_type != "Hybrid": 
            return await pipeline.run_async(data, include_outputs_from=["retriever"])
        else:
            return await pipeline.run_async(data, include_outputs_from=["joiner"])
    
    retrieval_methods = ["Keyword-based", "Embedding", "Hybrid"]
    
    task = []
    for pipeline, retrieval_method in zip(_pipelines, retrieval_methods):
        if retrieval_method == "Keyword-based":
            task.append((pipeline, {
        "retriever": {"query": question},
        "prompt_builder": {"question": question}
    }, retrieval_method))
        elif retrieval_method == "Embedding":
            task.append((pipeline, {
        "embedder": {"text": question},
        "prompt_builder": {"question": question}
    }, retrieval_method))
        elif retrieval_method == "Hybrid":
            task.append((pipeline, {
        "embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "prompt_builder": {"question": question}
    }, retrieval_method))
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    task = [run_pipe(*t) for t in task]
     
    pipe_results = loop.run_until_complete(asyncio.gather(*task))
    loop.close()

    keyword_answer = pipe_results[0]["generator"]["replies"][0]
    keyword_docs = pipe_results[0]["retriever"]["documents"]

    embedding_answer = pipe_results[1]["generator"]["replies"][0]
    embedding_docs = pipe_results[1]["retriever"]["documents"]

    hybrid_answer = pipe_results[2]["generator"]["replies"][0]
    hybrid_docs = pipe_results[2]["joiner"]["documents"]
        
    return keyword_answer, keyword_docs, embedding_answer, embedding_docs, hybrid_answer, hybrid_docs

@st.cache_data(show_spinner=True)
def prep_for_evaluation():
    import json
    with open("evaluation/eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
        return questions, answers # Replace with actual import

@st.cache_data(show_spinner=False)
def calculate_sas(ground_truth_answer, answer):
    sas_evaluator = SASEvaluator()
    sas_evaluator.warm_up()
    eval_result = sas_evaluator.run(ground_truth_answers=[ground_truth_answer], 
                                        predicted_answers=[answer])
    return eval_result["score"]


