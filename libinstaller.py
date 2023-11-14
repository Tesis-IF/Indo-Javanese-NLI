import subprocess
import sys

from logger.logger import init_logger

def installer(logger):
    try:
        import gdown
        logger.info("installer: gdown installed...")
    except ImportError:
        logger.info("installer: Installing gdown...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "gdown"])

    try:
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
        logger.info("installer: scikit-learn installed...")
    except ImportError:
        logger.info("installer: Installing scikit-learn...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', "scikit-learn"])

    try:
        import wandb
        logger.info("installer: wandb installed...")
    except ImportError:
        logger.info("installer: Installing wandb...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', "wandb"])

    try:
        from tqdm import tqdm, trange
        logger.info("installer: tqdm installed...")
    except ImportError:
        logger.info("installer: Installing tqdm...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "tqdm"])

    try:
        from decouple import config
        logger.info("installer: python-decouple installed...")
    except ImportError:    
        logger.info("installer: Installing python-decouple...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "python-decouple"])

    try:
        from transformers import AdamW, PreTrainedModel, PretrainedConfig, XLMRobertaModel, XLMRobertaTokenizer, BertTokenizer, BertModel
        logger.info("installer: transformers installed...")
    except ImportError:
        logger.info("installer: Installing transformers from huggingface...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "git+https://github.com/huggingface/transformers.git"])
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "git+https://github.com/huggingface/accelerate.git"])

    try:
        from huggingface_hub import login, logout
        logger.info("installer: huggingface_hub installed...")
    except ImportError:
        logger.info("installer: Installing huggingface_hub...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "huggingface_hub"])

    try:
        import torch
        logger.info("installer: pytorch installed...")
    except ImportError:
        logger.info("installer: Installing pytorch...")
        subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '-U', "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"])