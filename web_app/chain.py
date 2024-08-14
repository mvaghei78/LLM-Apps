"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import json

logger = logging.getLogger(__name__)


def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store

def load_prompt_template(prompt_path):
    """Load a prompt from a JSON file with system_template and human_template.
    Args:
        prompt_path (str): prompt file path
    Returns:
        ChatPromptTemplate: A ChatPromptTemplate object
    """
    with open(prompt_path, 'r') as file:
        prompt_data = json.load(file)
    if 'system_template' in prompt_data and 'human_template' in prompt_data:
        system_template = prompt_data['system_template']
        human_template = prompt_data['human_template']
        # Define the chat prompt using the system and human templates
        chat_prompt = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('human', human_template)
        ])
        return chat_prompt
    else:
        raise ValueError("The prompt JSON file does not contain the required keys 'system_template' and 'human_template'.")




def load_chain(wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=wandb_run.config.model_name,
        temperature=wandb_run.config.chat_temperature,
        max_retries=wandb_run.config.max_fallback_retries,
    )
    chat_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.chat_prompt_artifact, type="prompt"
    ).download()
    qa_prompt = load_prompt_template(f"{chat_prompt_dir}/prompt.json")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return qa_chain


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list[tuple[str, str]],
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """
    result = chain(
        inputs={"question": question, "chat_history": chat_history},
        return_only_outputs=True,
    )
    response = f"Answer:\t{result['answer']}"
    return response