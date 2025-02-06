
import streamlit as st
import logging
import os

from utils.haystack import start_document_store, start_haystack_pipeline, query_with_method, prep_for_evaluation, calculate_sas
from utils.ui import reset_results, set_initial_state, sidebar, pipeline_tab, haystack_tab, next_steps_tab

try:
    sidebar()
    st.write("# Practical RAG with OPEA and Haystack")
    tab1, tab2, tab3, tab4 = st.tabs(["Demo", "Haystack", "App Details", "Next Steps"])
    set_initial_state()
    
    with tab1:
        with st.spinner("🗄️ &nbsp;&nbsp; Indexing Files"):
            document_store = start_document_store()
        with st.spinner("🔀 &nbsp;&nbsp; Creating the pipelines"):    
            keyword_search_rag, embedding_rag, hybrid_rag = start_haystack_pipeline(document_store)
        example_questions, answers = prep_for_evaluation()
        st.markdown("""## 📑 Compare different retrieval methods \nKeyword-based, embedding, hybrid
                    """)
        # Search bar
        options = example_questions[:5]
        ground_truth_data = list(zip(example_questions[:5], answers[:5]))
        selection = st.pills("Example Questions", options, selection_mode="single")
        question = st.text_input("Ask a question", placeholder="Enter your query", value=selection or st.session_state.question, max_chars=100, on_change=reset_results)
        run_pressed = st.button("Run")
        ground_truth_answer = [item[1] for item in ground_truth_data if item[0] == question]
        if ground_truth_answer:
            st.markdown("""**Ground Truth Answer**:""")
            st.text(ground_truth_answer[0])
        run_query = (
            run_pressed and (question != st.session_state.question)
        )

        # Get results for query
        if run_query and question:
            reset_results()
            st.session_state.question = question
            col1, col2, col3 = st.columns(3)
            with st.spinner("🔎 &nbsp;&nbsp; Running three pipelines"):
                try:
                    keyword_answer, keyword_docs, embedding_answer, embedding_docs, hybrid_answer, hybrid_docs = query_with_method([keyword_search_rag, embedding_rag, hybrid_rag], question)
                except Exception as e:
                    logging.exception(e)
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
            with col1:
                st.markdown("""**Keyword-based Retrieval**:""")
                st.write(keyword_answer)
                # if ground_truth_answer:
                #     with st.spinner("Calculating SAS Score"):
                    # score = calculate_sas(ground_truth_answer[0], keyword_answer)
                    # st.markdown("""**SAS Score**:""")
                    # st.text(score)
                # if st.session_state.results:
                #     results = st.session_state.results
                #     st.write(results)
                #     if ground_truth_answer:
                #         with st.spinner("Calculating SAS Score"):
                #             score = calculate_sas(ground_truth_answer[0], results)
                #             st.markdown("""**SAS Score**:""")
                #             st.text(score)
                #     with st.expander("## Source Documents"):
                #         for doc in st.session_state.documents:
                #             st.markdown(doc.content)
            with col2:
                st.markdown("""**Embedding Retrieval**:""")
                st.write(embedding_answer)

            with col3:
                st.markdown("""**Hybrid Retrieval**:""")
                st.write(hybrid_answer)

            with st.spinner("🔎 &nbsp;&nbsp; Calculating SAS"):
                try:
                    keyword_sas = calculate_sas(ground_truth_answer[0], keyword_answer)
                    embedding_sas = calculate_sas(ground_truth_answer[0], embedding_answer)
                    hybrid_sas = calculate_sas(ground_truth_answer[0], hybrid_answer)
                except Exception as e:
                    logging.exception(e)
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
            

            with col1:
                st.markdown("""**SAS**""")
                st.text(keyword_sas)
            with col2:
                st.markdown("""**SAS**""")
                st.text(embedding_sas)
            with col3:
                st.markdown("""**SAS**""")
                st.text(hybrid_sas)
     
    with tab2:
        haystack_tab()
    
    with tab3:
        pipeline_tab()
        
    with tab4:
        next_steps_tab()
    
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)