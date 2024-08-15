# 🚀 LLM-Apps Repository

Welcome to the **LLM-Apps** repository! This repository contains all the code and resources from the [**Building LLM-Powered Apps**](https://www.wandb.courses/courses/building-llm-powered-apps) course provided by Weights & Biases (W&B). This course has been an incredible journey into the world of Large Language Models (LLMs), where I explored various applications, techniques, and tools to effectively harness the power of these models.

📄 You can view the full project report on the W&B website by following this link:

🔗 [LLM Apps Project Report](https://wandb.ai/mary1378/llmapps/reports/LLM-Apps-Project-Report--Vmlldzo5MDM1OTA0)

## 📂 Repository Structure

Here’s a quick overview of the directory structure within this repository:

- **`./artifacts/`**: 📦 Contains artifacts stored in W&B and downloaded for use.
- **`./files/`**: 🗂️ Files provided by W&B that are used in our code, saved here for easy access.
- **`./images/`**: 🖼️ Images used for better commenting and explanation.
- **`./jupyter_notebooks/`**: 📒 Jupyter notebook files containing the code and experiments.
- **`./vector_store/`**: 🧠 Stores the vector embeddings for document retrieval.
- **`./web_app/`**: 🌐 Python files for the web application, including the chatbot.
- **`./result/`**: 📝 Output result files generated from our code.

## 📊 Project Segments

The project is divided into several parts, each focusing on different aspects of working with LLMs:

### Part 1: Jupyter Notebook Files

1. **`Using_APIs.ipynb`**: 
   - Understand Tokenization, Temperature, Top_P, and the Chat API. 
   - 🌐 **Learn**: How to interact with LLMs through APIs.

2. **`Retrieval.ipynb`**: 
   - Tokenize and embed documents, retrieve the most relevant ones, and answer queries using the OpenAI model.
   - 🔍 **Focus**: Document retrieval and query answering.

3. **`Generation.ipynb`**: 
   - Dive deeper into prompting the model with better context using available user data and documentation files.
   - 📝 **Explore**: Advanced prompt engineering for better outputs.

### Part 2: Vector Store and Embedding

1. **`../web_app/ingest.py`**: 
   - Ingest a directory of documentation files into a vector store and store relevant artifacts in W&B.
   - 🧠 **Process**: Building and managing a vector store for efficient retrieval.

### Part 3: Web App Files

1. **`./web_app/app.py`**: 
   - Define a simple chatbot using LangChain and Gradio UI to answer questions about W&B documentation.
   - 🤖 **Build**: A conversational AI interface for user interaction.

2. **`./web_app/chain.py`**: 
   - Load a `ConversationalRetrievalChain` with vector store, prompt templates, and more to get answers from the chain.
   - 🔗 **Connect**: Chain components to create a seamless Q&A experience.

3. **`./web_app/config.py`**: 
   - Contains all the configurations needed for our web app.
   - ⚙️ **Configure**: Settings and parameters for optimal performance.

### Part 4: Evaluate LLM Output

1. **`./web_app/eval.py`**: 
   - Evaluate a `ConversationalRetrievalChain` on a dataset of questions and answers, using another LLM to score the results.
   - 📈 **Evaluate**: Performance and accuracy of the LLM's responses.

2. **`./web_app/prompt.py`**: 
   - Contains the system and human messages for our evaluator LLM.
   - 🧾 **Design**: Prompts and responses for evaluation and scoring.