"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = 'llmapps'
JOB_TYPE = 'production'

default_config = SimpleNamespace(
    project=PROJECT,
    entry=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact='mary1378/llmapps/vector_store:latest', # Where do we pull our vector store
    chat_prompt_artifact='mary1378/llmapps/chat_prompt:latest', # Where do we pull our prompt template
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name='gpt-3.5-turbo',
    eval_model='gpt-3.5-turbo',
    eval_artifact='mary1378/llmapps/generated_examples:v0'
)