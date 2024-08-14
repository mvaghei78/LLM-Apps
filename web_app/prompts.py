"""Prompts for the chatbot and evaluation."""
import json
import logging
import pathlib
from typing import Union

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

logger = logging.getLogger(__name__)

def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template  = f_name.open('r').read()
    else:
        logger.warning(
            f'No human prompt provided. Using default human prompt from {__name__}'
        )
        human_template = """\nQUESTION: {query}\nCHATBOT ANSWER:{result}\n
        ORIGINAL ANSWER: {answer}\nGRADE:"""
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an evaluator for the W&B chatbot. You are given a question, the chatbot answer, and the original answer,
        and are asked to score the chatbot's answer as either CORRECT or INCORRECT. Note
        that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the 
        best answer. You are evaluating the chatbot's answer only. Example format:\nQUESTION: question here\nCHATBOT
        ANSWER: student's answer here\nORIGINAL ANSWER: original answer here\nGRADE: CORRCT or INCORRECT here\nPlease 
        remember to grade them based on being factually accurate. Begin!"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt