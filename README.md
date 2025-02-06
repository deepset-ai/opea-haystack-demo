---
title: Haystack Application with Streamlit
emoji: 👑
colorFrom: indigo
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

# Demoplooza: Practical RAG with OPEA and Haystack

This template [Streamlit](https://docs.streamlit.io/) app is set up for simple [Haystack](https://haystack.deepset.ai/) applications. The template is ready to do **Retrievel Augmented Generation** on example files.

See the ['How to use this template'](#how-to-use-this-template) instructions below to create a simple UI for your own Haystack search pipelines.

## Installation and Running
To run the bare application:
1. Install requirements: `pip install -r requirements.txt`
2. Make sure that your OPEA microservices are running
3. Run the streamlit app: `streamlit run app.py`

This will start up the app on `localhost:8501` where you will find a simple search bar. 

## How to use this template
1. Create a new repository from this template or simply open it in a codespace to start playing around 💙
2. Make sure your `requirements.txt` file includes the Haystack (`haystack-ai`) and Streamlit versions you would like to use.
3. Change the code in `utils/haystack.py` if you would like a different pipeline. 
4. Create a `.env` file with all of your configuration settings.
5. Make any UI edits if you'd like to.
6. Run the app as show in [installation and running](#installation-and-running)

### Repo structure
- `./utils`: This is where we have 2 files: 
    - `haystack.py`: Here you will find some functions already set up for you to start creating your Haystack search pipeline. It includes 2 main functions called `start_haystack_pipeline()` which is what we use to create a pipeline and cache it, and `query()` which is the function called by `app.py` once a user query is received.
    - `ui.py`: Use this file for any UI and initial value setups.
- `app.py`: This is the main Streamlit application file that we will run. In its current state it has a sidebar, a simple search bar, a 'Run' button, and a response.
- `./files`: You can use this folder to store files to be indexed.

### What to edit?
There are default pipelines both in `start_document_store()` and `start_haystack_pipeline()`. Change the pipelines to use different document stores, embedding and generative models or update the pipelines as you need. Check out [📚 Useful Resources](#-useful-resources) section for details.

### 📚 Useful Resources
* [Get Started](https://haystack.deepset.ai/overview/quick-start)
* [Docs](https://docs.haystack.deepset.ai/docs/intro)
* [Tutorials](https://haystack.deepset.ai/tutorials)
* [Integrations](https://haystack.deepset.ai/integrations)
