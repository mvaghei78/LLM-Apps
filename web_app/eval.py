"""Evaluate a ConversationalRetrievalChain on a dataset of questions and answers."""
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import wandb
from chain import load_chain, load_vector_store
from config import default_config
from langchain.chains import ConversationalRetrievalChain
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from prompts import load_eval_prompt
from tqdm import tqdm

def load_eval_dataset(config: SimpleNamespace) -> pd.DataFrame:
    """Load a dataset of questions and answers from a Weights & Biases artifact
    Args:
        config (SimpleNamespace): a config object
    Returns:
        pd.DataFrame: A dataframe of questions and answers
    """
    # we will load data from a wandb Table artifact
    artifact = wandb.use_artifact(config.eval_artifact)
    # download artifact
    artifact_dir = Path(artifact.download())
    # load data
    eval_dataset = pd.read_csv(artifact_dir / 'generated_examples.csv')
    return eval_dataset

def generate_answers(eval_dataset: pd.DataFrame, qa_chain: ConversationalRetrievalChain, results_dir: str) -> pd.DataFrame:
    """Generate answers for a dataset of questions and answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions and answers
        qa_chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        results_dir (str): The directory that we use to store our output results
    Returns:
        pd.DataFrame: A dataframe of questions, answers, and model answers
    """
    answers = []
    for query in tqdm(eval_dataset['question'], total=len(eval_dataset)):
        result = qa_chain({'question': query, 'chat_history': []})
        answers.append(result['answer'])

    eval_dataset['model_answer'] = answers
    eval_dataset.to_csv(results_dir+'/eval_with_answers.csv', index=False)
    return eval_dataset

def evaluate_answers(eval_dataset: pd.DataFrame, config: SimpleNamespace) -> pd.DataFrame:
    """Evaluate a dataset of questions, answers, and model answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions,answers, and model answers
        config (SimpleNamespace): A config object
    Returns:
        pd.DataFrame: A dataframe of questions, answers, model answers, and model scores
    """
    eval_prompt = load_eval_prompt()
    llm = ChatOpenAI(
        model_name=config.eval_model,
        temperature=0
    )
    eval_chain = QAEvalChain.from_llm(llm, prompt=eval_prompt)
    
    examples = []
    predictions = []
    for i in range(len(eval_dataset)):
        examples.append({
            'query': eval_dataset['question'].iloc[i],
            'answer': eval_dataset['answer'].iloc[i]
        })
        predictions.append({
            'query': eval_dataset['question'].iloc[i],
            'answer': eval_dataset['answer'].iloc[i],
            'result': eval_dataset['model_answer'].iloc[i]
        })
    graded_outputs = eval_chain.evaluate(examples, predictions)
    # Initialize 'model_score' column
    eval_dataset['model_score'] = ''

    for idx, x in enumerate(graded_outputs):
        result = x['results']
        if "GRADE: CORRECT" in result:
            eval_dataset.at[idx, 'model_score'] = 'CORRECT'
        elif "GRADE: INCORRECT" in result:
            eval_dataset.at[idx, 'model_score'] = 'INCORRECT'
        else:
            eval_dataset.at[idx, 'model_score'] = 'None'

    return eval_dataset

def log_results(eval_dataset: pd.DataFrame, results_dir: str) -> None:
    """Log evaluation results to a Weights & Biases Artifact
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions, answers, model_answers, and model scores
        results_dir (str): The directory that we use to store our output results
    """
    model_accuracy = len(eval_dataset[eval_dataset['model_score'] == 'CORRECT']) / len(eval_dataset)
    wandb.log({'model_accuracy': model_accuracy})
    eval_dataset.to_csv(results_dir+'/eval_results.csv', index=False)
    artifact = wandb.Artifact('eval_results', type='eval_results')
    artifact.add_file(results_dir+'/eval_results.csv')
    wandb.log_artifact(artifact)
    wandb.log({'eval_results': wandb.Table(dataframe=eval_dataset)})

if __name__ == '__main__':
    results_dir = '../result'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    with wandb.init(project=default_config.project, config=default_config, job_type='eval') as run:
        eval_dataset = load_eval_dataset(default_config)
        vector_store = load_vector_store(run, os.environ['OPENAI_API_KEY'])
        qa_chain = load_chain(run, vector_store, os.environ['OPENAI_API_KEY'])
        eval_dataset = generate_answers(eval_dataset, qa_chain)
        # eval_dataset = pd.read_csv(results_dir+'/eval_with_answers.csv')
        eval_dataset = evaluate_answers(eval_dataset, default_config)
        log_results(eval_dataset, results_dir)
        wandb.finish()