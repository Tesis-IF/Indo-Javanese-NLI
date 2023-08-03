#!/usr/bin/env python
# coding: utf-8

# In[263]:


get_ipython().system(' pip install -U scikit-learn')
get_ipython().system(' pip install wandb')
get_ipython().system(' pip install tqdm')
get_ipython().system(' pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117')
get_ipython().system(' pip install -U git+https://github.com/huggingface/transformers.git')
get_ipython().system(' pip install -U git+https://github.com/huggingface/accelerate.git')


# In[1]:


from calendar import EPOCH
import pandas as pd
import numpy as np
import os
import gc
import random
import time
from tqdm import tqdm, trange

import gdown

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set a seed value
torch.manual_seed(42)

import wandb

import transformers
from transformers import AdamW
from transformers import PreTrainedModel, PretrainedConfig
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from huggingface_hub import login


# In[2]:


TOKENIZER_TYPE = 'xlm-roberta-base'
MBERT_TYPE = 'xlm-roberta-base'
MODEL_TEACHER_TYPE = 'jalaluddin94/xlmr-nli-indoindo'
HF_MODEL_NAME = 'jalaluddin94/trf-learning-indojavanesenli-xlmr'

# DATASET_NAME = 'jalaluddin94/IndoJavaneseNLI'

STUDENT_LRATE = 2e-5
LAMBDA_KLD = 0.5 # between 0.01 - 0.5
MAX_LEN = 512
NUM_EPOCHS = 3
BATCH_SIZE = 8
BATCH_NORM_EPSILON = 1e-5
LAMBDA_L2 = 3e-5

HF_TOKEN = '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'

NUM_CORES = os.cpu_count() - 2

print(f"Percobaan 1 - Epoch {EPOCH} Learning, Batch size {BATCH_SIZE}, Rate Student {STUDENT_LRATE}, Lambda KLD: {LAMBDA_KLD}")

# In[3]:


# Preparing dataset
print("Preparing dataset...")
if not os.path.exists("datasets/"):
  os.makedirs("datasets/")

# Download dataset
print("Downloading training data...")
uri = "https://drive.google.com/uc?id=1j5iclahnkuk_jCZS12ZB9a4QN_XAJ9pt"
output = "datasets/train.csv"
gdown.download(url=uri, output=output, quiet=False, fuzzy=True)

print("Downloading validation data...")
uri = "https://drive.google.com/uc?id=1A4M8uS3bl__-Jugq11cOCAXdvXdMKBju"
output = "datasets/validation.csv"
gdown.download(url=uri, output=output, quiet=False, fuzzy=True)

print("Downloading testing data...")
uri = "https://drive.google.com/uc?id=1h011UmkFi9gM1yGEicizrGAUxgmPI1TP"
output = "datasets/test.csv"
gdown.download(url=uri, output=output, quiet=False, fuzzy=True)


# In[4]:


login(token=HF_TOKEN)


# In[266]:


get_ipython().run_line_magic('env', 'WANDB_API_KEY=97b170d223eb55f86fe1fbf9640831ad76381a74')
wandb.login()


# In[267]:


get_ipython().run_line_magic('env', "WANDB_LOG_MODEL='end'")
run = wandb.init(
  project="javanese_nli",
  notes="Experiment transfer learning on Bandyopadhyay's paper using XLMR",
  name="trf-lrn-experiment-xlmr-epoch3-batchsize8-lamdakld0.5",
  tags=["transferlearning", "bandyopadhyay", "xlmr"]
)


# In[268]:


os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]="1024"
os.environ["WANDB_AGENT_DISABLE_FLAPPING"]="true"


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Data Preparation

# Prepare Dataset for Student

# In[9]:


df_train = pd.read_csv("datasets/train.csv", sep='\t')
df_train = df_train.sample(frac=1).reset_index(drop=True) #shuffle the data

df_train_student = pd.DataFrame()
df_train_student["premise"] = df_train["premise"]
df_train_student["hypothesis"] = df_train["jv_hypothesis_mongo"]
df_train_student["label"] = df_train["label"]
df_train_student.head()


# In[10]:


df_valid = pd.read_csv("datasets/validation.csv", sep='\t')
df_valid = df_valid.sample(frac=1).reset_index(drop=True) #shuffle the data

df_valid_student = pd.DataFrame()
df_valid_student["premise"] = df_valid["premise"]
df_valid_student["hypothesis"] = df_valid["jv_hypothesis_mongo"]
df_valid_student["label"] = df_valid["label"]
df_valid_student.head()


# In[11]:


df_test = pd.read_csv("datasets/test.csv", sep='\t')
df_test = df_test.sample(frac=1).reset_index(drop=True) #shuffle the data

df_test_student = pd.DataFrame()
df_test_student["premise"] = df_test["premise"]
df_test_student["premise"] = df_test_student["premise"].astype(str)
df_test_student["hypothesis"] = df_test["jv_hypothesis"]
df_test_student["hypothesis"] = df_test_student["hypothesis"].astype(str)
df_test_student["label"] = df_test["label"]
df_test_student.head()


# Prepare Dataset for Teacher

# Dataset from teacher will be from "IndoNLI", and using Indonesian only.

# In[12]:


df_train_t = pd.DataFrame()
df_train_t["premise"] = df_train["premise"]
df_train_t["hypothesis"] = df_train["hypothesis"]
df_train_t["label"] = df_train["label"]
df_train_t = df_train_t.sample(frac=1).reset_index(drop=True)
display(df_train_t)


# In[13]:


print("Count per class train:") 
print(df_train_t['label'].value_counts())


# In[14]:


df_valid_t = pd.DataFrame()
df_valid_t["premise"] = df_valid["premise"]
df_valid_t["hypothesis"] = df_valid["hypothesis"]
df_valid_t["label"] = df_valid["label"]
df_valid_t = df_valid_t.sample(frac=1).reset_index(drop=True)
display(df_valid_t)


# In[15]:


print("Count per class valid:") 
print(df_valid_t['label'].value_counts())


# In[16]:


df_test_t = pd.DataFrame()
df_test_t["premise"] = df_test["premise"]
df_test_t["hypothesis"] = df_test["hypothesis"]
df_test_t["label"] = df_test["label"]
df_test_t = df_test_t.sample(frac=1).reset_index(drop=True)
display(df_test_t)


# In[17]:


print("Count per class test:") 
print(df_test_t['label'].value_counts())


# ## Preprocessing

# In[18]:


tokenizer = XLMRobertaTokenizer.from_pretrained(TOKENIZER_TYPE)


# In[19]:


class CompDataset(Dataset):
    def __init__(self, df_teacher, df_student):
        self.df_data_teacher = df_teacher
        self.df_data_student = df_student
        
    def __getitem__(self, index):
        # Teacher
        sentence_teacher_1 = self.df_data_teacher.loc[index, 'premise']
        sentence_teacher_2 = self.df_data_teacher.loc[index, 'hypothesis']
        
        encoded_dict_teacher = tokenizer.encode_plus(
            sentence_teacher_1,
            sentence_teacher_2,
            add_special_tokens = True,
            max_length = MAX_LEN,
            truncation='longest_first',
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        padded_token_list_teacher = encoded_dict_teacher['input_ids'][0]
        att_mask_teacher = encoded_dict_teacher['attention_mask'][0]
        
        target_teacher = torch.tensor([self.df_data_teacher.loc[index, 'label']])
        lt_target_teacher = torch.LongTensor(target_teacher)
        onehot_encoded_lbl_teacher = F.one_hot(lt_target_teacher, num_classes=3) # 3 classes: entails, neutral, contradict
        
        # Student
        sentence_student_1 = self.df_data_student.loc[index, 'premise']
        sentence_student_2 = self.df_data_student.loc[index, 'hypothesis']
        
        encoded_dict_student = tokenizer.encode_plus(
            sentence_student_1,
            sentence_student_2,
            add_special_tokens = True,
            max_length = MAX_LEN,
            truncation='longest_first',
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        padded_token_list_student = encoded_dict_student['input_ids'][0]
        att_mask_student = encoded_dict_student['attention_mask'][0]
        
        target_student = torch.tensor([self.df_data_student.loc[index, 'label']])
        lt_target_student = torch.LongTensor(target_student)
        onehot_encoded_lbl_student = F.one_hot(lt_target_student, num_classes=3) # 3 classes: entails, neutral, contradict
        
        output = {
            "input_ids_teacher": padded_token_list_teacher, 
            "attention_mask_teacher": att_mask_teacher,
            "lbl_teacher": onehot_encoded_lbl_teacher,
            "input_ids_student": padded_token_list_student, 
            "attention_mask_student": att_mask_student,
            "lbl_student": onehot_encoded_lbl_student
        }
        
        return output
    
    def __len__(self):
        return len(self.df_data_teacher)


# In[281]:


train_data_cmp = CompDataset(df_train_t, df_train_student)
valid_data_cmp = CompDataset(df_valid_t, df_valid_student)
test_data_cmp = CompDataset(df_test_t, df_test_student)


# In[282]:


train_dataloader = DataLoader(train_data_cmp, batch_size = BATCH_SIZE)
valid_dataloader = DataLoader(valid_data_cmp, batch_size = BATCH_SIZE)
test_dataloader = DataLoader(test_data_cmp, batch_size = BATCH_SIZE)


# ## Model

# Transfer Learning model as per Bandyopadhyay, D., et al (2022) paper, but using XLMR instead of mBERT

# In[283]:


class TransferLearningPaper(PreTrainedModel):
    def __init__(self, config, lambda_kld, learningrate_student, batchnorm_epsilon = 1e-5):
        super(TransferLearningPaper, self).__init__(config)
        
        self.xlmr_model_teacher = XLMRobertaModel.from_pretrained(
            MODEL_TEACHER_TYPE, # using pretrained mBERT in INA language
            num_labels = 3,
            output_hidden_states=True
        )
        
        # Freeze teacher mBERT parameters
        for params_teacher in self.xlmr_model_teacher.parameters():
            params_teacher.requires_grad = False
    
        self.xlmr_model_student = XLMRobertaModel.from_pretrained(
            MBERT_TYPE,
            num_labels = 3,
            output_hidden_states=True
        )
        
        # Unfreeze student mBERT parameters
        for params_student in self.xlmr_model_student.parameters():
            params_student.requires_grad = True
        
        self.optimizer_student = AdamW(
            self.xlmr_model_student.parameters(), 
            lr=learningrate_student
        )
        
        self.linear = nn.Linear(config.hidden_size, 3)  # Linear layer
        self.batchnorm = nn.BatchNorm1d(config.hidden_size, eps=batchnorm_epsilon)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        
        # Initialize the weights of the linear layer
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        
        self.lambda_kld = lambda_kld
    
    def forward(self, input_ids_teacher, attention_mask_teacher, lbl_teacher, input_ids_student, attention_mask_student, lbl_student):
        # the label is already one-hot encoded 
        self.xlmr_model_teacher.eval()
        self.xlmr_model_student.eval()
        
        lbl_teacher = lbl_teacher[:, 0, :]
        lbl_student = lbl_student[:, 0, :]
        
        with torch.no_grad():
            # Taking CLS token out of XLMR last hidden state
            outputs_teacher = self.xlmr_model_teacher(
                input_ids=input_ids_teacher, 
                attention_mask=attention_mask_teacher
            )
        
            # take CLS token of the last hidden state
            pooled_output_teacher = outputs_teacher.last_hidden_state[:, 0, :]
        
        # taking CLS token out of the student data without deleting the gradient
        outputs_student = self.xlmr_model_student(
            input_ids=input_ids_student, 
            attention_mask=attention_mask_student 
        )
        
        pooled_output_student = outputs_student.last_hidden_state[:, 0, :]
        
        # FFNN
        batchnormed_logits = self.batchnorm(pooled_output_student)
        linear_output = self.linear(batchnormed_logits) # the output's logits
        softmax_linear_output = F.log_softmax(linear_output, dim=1)
        
        lbl_student = lbl_student.float()
        softmax_linear_output = softmax_linear_output.float()
        
        # Loss Computation
        cross_entropy_loss = self.cross_entropy(softmax_linear_output, lbl_student)
        total_kld = self.kld(F.log_softmax(pooled_output_student, dim=1), F.softmax(pooled_output_teacher, dim=1))
        joint_loss = cross_entropy_loss + (self.lambda_kld * total_kld )
        
        return {"loss": joint_loss, "logits": softmax_linear_output}
    
    def clear_grad(self):
        self.xlmr_model_student.train()
        self.optimizer_student.zero_grad()
    
    def backpro_compute(self, loss):
        loss.backward()
        
    def update_std_weights_and_clear_grad(self):
        self.optimizer_student.step()
        self.optimizer_student.zero_grad()
    
    def update_std_weights(self):
        self.optimizer_student.step()
    
    def update_param_student_model(self, loss):
        # Doing customized backpropagation for student's model
        self.xlmr_model_student.train()
        
        self.optimizer_student.zero_grad()
        loss.backward()
        self.optimizer_student.step()
        
    def upload_to_huggingface(self):
        self.xlmr_model_student.push_to_hub(HF_MODEL_NAME)
        tokenizer.push_to_hub(HF_MODEL_NAME)


# In[284]:


config = PretrainedConfig(
    problem_type = "single_label_classification",
    id2label = {
        "0": "ENTAIL",
        "1": "NEUTRAL",
        "2": "CONTRADICTION"
    },
    label2id = {
        "ENTAIL": 0,
        "NEUTRAL": 1,
        "CONTRADICTION": 2
    },
    num_labels = 3,
    hidden_size = 768,
    name_or_path = "indojavanesenli-transfer-learning",
    finetuning_task = "indonesian-javanese natural language inference"
)
print(config)
transferlearning_model = TransferLearningPaper(
    config = config,
    lambda_kld = LAMBDA_KLD, # antara 0.01-0.5
    learningrate_student = STUDENT_LRATE,
    batchnorm_epsilon = BATCH_NORM_EPSILON
)
transferlearning_model = transferlearning_model.to(device)


# ## Training

# In[ ]:


gc.collect()


# Function to compute metrics

# In[ ]:


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    labels = np.argmax(labels[:,0,:], axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


# Manual training function

# In[ ]:


def train(the_model, train_data, pgb):
    the_model.train()
    
    batch_loss = 0
    
    for batch, data in enumerate(train_data):
        # Clear accumulated gradients
        the_model.clear_grad()
        
        input_ids_teacher = data["input_ids_teacher"].to(device)
        attention_mask_teacher = data["attention_mask_teacher"].to(device)
        lbl_teacher = data["lbl_teacher"].to(device)
        input_ids_student = data["input_ids_student"].to(device)
        attention_mask_student = data["attention_mask_student"].to(device)
        lbl_student = data["lbl_student"].to(device)
        
        output = the_model(
            input_ids_teacher = input_ids_teacher, 
            attention_mask_teacher = attention_mask_teacher,
            lbl_teacher = lbl_teacher,
            input_ids_student = input_ids_student, 
            attention_mask_student = attention_mask_student, 
            lbl_student = lbl_student
        )
        
        loss_model = output["loss"]
        batch_loss += loss_model
        
        # Backpropagation
        # the_model.update_param_student_model(loss_model) # uncomment to use ordinary backpro
        ## now using gradient accumulation technique
        the_model.backpro_compute(loss_model) # backward pass and gradient accumulation
        
        # Accumulate gradients for the desired number of mini-batches
        if(batch+1) % BATCH_SIZE == 0:
            # update weights
            the_model.update_std_weights_and_clear_grad()
        
        pgb.update(1 / len(train_data))
    
    # Make sure to update the weights for any remaining accumulated gradients
    if (batch+1) % BATCH_SIZE != 0:
        the_model.update_std_weights()
        
    training_loss = batch_loss / BATCH_SIZE
    wandb.log({"train/loss": training_loss})
    
    return training_loss


# In[ ]:


def validate(the_model, valid_data):
    the_model.eval()
    
    batch_loss = 0
    
    eval_f1 = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []
    
    with torch.no_grad():
        for batch, data in enumerate(valid_data):
            input_ids_teacher = data["input_ids_teacher"].to(device)
            attention_mask_teacher = data["attention_mask_teacher"].to(device)
            lbl_teacher = data["lbl_teacher"].to(device)
            input_ids_student = data["input_ids_student"].to(device)
            attention_mask_student = data["attention_mask_student"].to(device)
            lbl_student = data["lbl_student"].to(device)

            output = the_model(
                input_ids_teacher = input_ids_teacher, 
                attention_mask_teacher = attention_mask_teacher, 
                lbl_teacher = lbl_teacher,
                input_ids_student = input_ids_student, 
                attention_mask_student = attention_mask_student,
                lbl_student = lbl_student
            )

            logits = output["logits"].cpu().detach().numpy()
            packed_val = logits, lbl_student.cpu().detach().numpy()
            metrics = compute_metrics(packed_val)
            
            eval_f1.append(metrics["f1_score"])
            eval_accuracy.append(metrics["accuracy"])
            eval_precision.append(metrics["precision"])
            eval_recall.append(metrics["recall"])
            
            loss_model = output["loss"]
            batch_loss += loss_model
    
        eval_loss = batch_loss / BATCH_SIZE
        wandb.log({
            "eval/loss": eval_loss, 
            "eval/f1_score": np.average(eval_f1), 
            "eval/accuracy": np.average(eval_accuracy),
            "eval/precision": np.average(eval_precision),
            "eval/recall": np.average(eval_recall)
        })
    
    out_metrics = {
        "eval/loss": eval_loss, 
        "eval/f1_score": np.average(eval_f1), 
        "eval/accuracy": np.average(eval_accuracy),
        "eval/precision": np.average(eval_precision),
        "eval/recall": np.average(eval_recall)
    }
    
    return eval_loss, out_metrics


# In[ ]:


def training_sequence(the_model, train_data, valid_data, epochs):
    track_train_loss = []
    track_val_loss = []
    
    pbar_format = "{l_bar}{bar} | Epoch: {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"
    with tqdm(total=epochs, colour="blue", leave=True, position=0, bar_format=pbar_format) as t:
        for ep in range(epochs):
            training_loss = train(the_model, train_data, t)
            t.set_description(f"Evaluating... Train loss: {training_loss:.3f}")
            valid_loss, _ = validate(the_model, valid_data)

            track_train_loss.append(training_loss)
            track_val_loss.append(valid_loss)

            t.set_description(f"Train loss: {training_loss:.3f} Valid loss: {valid_loss:.3f}")

            if valid_loss < min(track_val_loss) or ep + 1 == 1:
                the_model.save_pretrained(
                    save_directory = MODEL_PATH + "indojavanesenli-transfer-learning"
                )

            wandb.log({
                "train_loss/epoch": training_loss,
                "validation_loss/epoch": valid_loss
            })
        
    return {
        "training_loss": track_train_loss,
        "validation_loss": track_val_loss
    }


# In[ ]:


training_result = training_sequence(transferlearning_model, train_dataloader, valid_dataloader, NUM_EPOCHS)


# In[ ]:


wandb.finish()


# In[ ]:

transferlearning_model.upload_to_huggingface()


# In[ ]:
STUDENT_LRATE = 2e-5
LAMBDA_KLD = 0.5 # between 0.01 - 0.5
MAX_LEN = 512
NUM_EPOCHS = 5
BATCH_SIZE = 8
BATCH_NORM_EPSILON = 1e-5
LAMBDA_L2 = 3e-5

print(f"Percobaan 2 - Epoch {EPOCH} Learning, Batch size {BATCH_SIZE}, Rate Student {STUDENT_LRATE}, Lambda KLD: {LAMBDA_KLD}")


# In[ ]:

run = wandb.init(
  project="javanese_nli",
  notes="Experiment transfer learning on Bandyopadhyay's paper using XLMR",
  name="trf-lrn-experiment-xlmr-epoch5-batchsize8-lamdakld0.5",
  tags=["transferlearning", "bandyopadhyay", "xlmr"]
)

# In[ ]:

transferlearning_model = TransferLearningPaper(
    config = config,
    lambda_kld = LAMBDA_KLD, # antara 0.01-0.5
    learningrate_student = STUDENT_LRATE,
    batchnorm_epsilon = BATCH_NORM_EPSILON
)

transferlearning_model = transferlearning_model.to(device)
training_result = training_sequence(transferlearning_model, train_dataloader, valid_dataloader, NUM_EPOCHS)

wandb.finish()
transferlearning_model.upload_to_huggingface()


# In[ ]:
STUDENT_LRATE = 2e-5
LAMBDA_KLD = 0.5 # between 0.01 - 0.5
MAX_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 8
BATCH_NORM_EPSILON = 1e-5
LAMBDA_L2 = 3e-5

print(f"Percobaan 3 - Epoch {EPOCH} Learning, Batch size {BATCH_SIZE}, Rate Student {STUDENT_LRATE}, Lambda KLD: {LAMBDA_KLD}")


# In[ ]:

run = wandb.init(
  project="javanese_nli",
  notes="Experiment transfer learning on Bandyopadhyay's paper using XLMR",
  name="trf-lrn-experiment-xlmr-epoch10-batchsize8-lamdakld0.5",
  tags=["transferlearning", "bandyopadhyay", "xlmr"]
)

# In[ ]:

transferlearning_model = TransferLearningPaper(
    config = config,
    lambda_kld = LAMBDA_KLD, # antara 0.01-0.5
    learningrate_student = STUDENT_LRATE,
    batchnorm_epsilon = BATCH_NORM_EPSILON
)

transferlearning_model = transferlearning_model.to(device)
training_result = training_sequence(transferlearning_model, train_dataloader, valid_dataloader, NUM_EPOCHS)

wandb.finish()
transferlearning_model.upload_to_huggingface()

# In[ ]:


logout()

