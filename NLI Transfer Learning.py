#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning Approach for Cross-Lingual NLI

# ## Import Libraries and Setup Environment Variables

# In[1]:


import pandas as pd
import numpy as np
import os
import gc
import random
import gdown
import time
from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set a seed value
torch.manual_seed(205)

from datasets import load_dataset

import wandb

import transformers
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from transformers import PreTrainedModel, PretrainedConfig
from transformers import BertTokenizer, BertModel, BertForSequenceClassification #, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW


# In[2]:


# MODEL_TYPE = 'xlm-roberta-base'
TOKENIZER_TYPE = 'bert-base-multilingual-cased'
MBERT_TYPE = 'bert-base-multilingual-cased'
MODEL_TYPE = 'jalaluddin94/nli_mbert'
MODEL_PATH = 'D:/Training/Machine Learning/NLP/NLI/saved_models/Indo-Javanese-NLI/ResearchedModels/'

L_RATE = 3e-6
STUDENT_LRATE = 3e-6
MAX_LEN = 512
NUM_EPOCHS = 25
BATCH_SIZE = 8
BATCH_NORM_EPSILON = 1e-5
LAMBDA_L2 = 3e-5

HF_TOKEN = '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'

NUM_CORES = os.cpu_count() - 2


# In[3]:


# %env WANDB_NOTEBOOK_NAME=/home/sagemaker-user/PPT/BERT_BiLSTM_Game_Review.ipynb
get_ipython().run_line_magic('env', 'WANDB_API_KEY=97b170d223eb55f86fe1fbf9640831ad76381a74')
wandb.login()


# In[4]:


# %env WANDB_PROJECT=javanese_nli
get_ipython().run_line_magic('env', "WANDB_LOG_MODEL='end'")
run = wandb.init(
  project="javanese_nli",
  notes="Experiment transfer learning on Bandyopadhyay's paper",
  name="transfer-learning-paper",
  tags=["transferlearning", "bandyopadhyay"]
)


# In[5]:


os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]="1024"


# In[6]:


os.environ["WANDB_AGENT_DISABLE_FLAPPING"]="true"


# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Download and Prepare Dataset

# ### Download Dataset

# In[8]:


# uri = "https://drive.google.com/uc?id=1aE9w2rqgW-j3PTgjnmHDjulNwp-Znb6i"
# output = "dataset/indo_java_nli_training.csv"
# if not os.path.exists("dataset/"):
#   os.makedirs("dataset/")
# gdown.download(url=uri, output=output, quiet=False, fuzzy=True)


# In[9]:


# uri = "https://drive.google.com/uc?id=1YlQ9_8CvQbTSb5-2BjIfiYT-cy7pe6YM"
# output = "dataset/indo_java_nli_validation.csv"
# if not os.path.exists("dataset/"):
#   os.makedirs("dataset/")
# gdown.download(url=uri, output=output, quiet=False, fuzzy=True)


# In[10]:


# uri = "https://drive.google.com/uc?id=1Zz_rHeI7fPUuA04zt9gCWyl5RYhrYPn0"
# output = "dataset/indo_java_nli_testing.csv"
# if not os.path.exists("dataset/"):
#   os.makedirs("dataset/")
# gdown.download(url=uri, output=output, quiet=False, fuzzy=True)


# ### Prepare Dataset for Student 

# In[11]:


df_train = pd.read_csv("D:/Training/Machine Learning/Datasets/NLI/IndoJavaNLI/indojavanesenli-train.csv", sep='\t')
df_train = df_train.sample(frac=1).reset_index(drop=True) #shuffle the data


# In[12]:


df_train_student = pd.DataFrame()
df_train_student["premise"] = df_train["premise"]
df_train_student["hypothesis"] = df_train["jv_hypothesis_mongo"]
df_train_student["label"] = df_train["label"]
df_train_student.head()


# In[13]:


df_valid = pd.read_csv("D:/Training/Machine Learning/Datasets/NLI/IndoJavaNLI/indojavanesenli-valid.csv", sep='\t')
df_valid = df_valid.sample(frac=1).reset_index(drop=True) #shuffle the data


# In[14]:


df_valid_student = pd.DataFrame()
df_valid_student["premise"] = df_valid["premise"]
df_valid_student["hypothesis"] = df_valid["jv_hypothesis_mongo"]
df_valid_student["label"] = df_valid["label"]
df_valid_student.head()


# In[15]:


df_test = pd.read_csv("D:/Training/Machine Learning/Datasets/NLI/IndoJavaNLI/indojavanesenli-test.csv", sep='\t')
df_test = df_test.sample(frac=1).reset_index(drop=True) #shuffle the data


# In[16]:


df_test_student = pd.DataFrame()
df_test_student["premise"] = df_test["premise"]
df_test_student["premise"] = df_test_student["premise"].astype(str)
df_test_student["hypothesis"] = df_test["jv_hypothesis_mongo"]
df_test_student["hypothesis"] = df_test_student["hypothesis"].astype(str)
df_test_student["label"] = df_test["label"]
df_test_student.head()


# ### Prepare Dataset for Teacher

# Dataset from teacher will be from "IndoNLI", and using Indonesian only.

# In[17]:


df_train_t = pd.DataFrame()
df_train_t["premise"] = df_train["premise"]
df_train_t["hypothesis"] = df_train["hypothesis"]
df_train_t["label"] = df_train["label"]
df_train_t = df_train_t.sample(frac=1).reset_index(drop=True)
display(df_train_t)


# In[18]:


print("Count per class train:") 
print(df_train_t['label'].value_counts())


# In[19]:


df_valid_t = pd.DataFrame()
df_valid_t["premise"] = df_valid["premise"]
df_valid_t["hypothesis"] = df_valid["hypothesis"]
df_valid_t["label"] = df_valid["label"]
df_valid_t = df_valid_t.sample(frac=1).reset_index(drop=True)
display(df_valid_t)


# In[20]:


print("Count per class valid:") 
print(df_valid_t['label'].value_counts())


# In[21]:


df_test_t = pd.DataFrame()
df_test_t["premise"] = df_test["premise"]
df_test_t["hypothesis"] = df_test["hypothesis"]
df_test_t["label"] = df_test["label"]
df_test_t = df_test_t.sample(frac=1).reset_index(drop=True)
display(df_test_t)


# In[22]:


print("Count per class test:") 
print(df_test_t['label'].value_counts())


# ## Preprocessing

# ### Tokenization

# In[23]:


# tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_TYPE)


# In[24]:


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
        tok_type_id_teacher = encoded_dict_teacher['token_type_ids'][0]
        
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
        tok_type_id_student = encoded_dict_student['token_type_ids'][0]
        
        target_student = torch.tensor([self.df_data_student.loc[index, 'label']])
        lt_target_student = torch.LongTensor(target_student)
        onehot_encoded_lbl_student = F.one_hot(lt_target_student, num_classes=3) # 3 classes: entails, neutral, contradict
        
        output = {
            "input_ids_teacher": padded_token_list_teacher, 
            "attention_mask_teacher": att_mask_teacher,
            "token_type_ids_teacher": tok_type_id_teacher,
            "lbl_teacher": onehot_encoded_lbl_teacher,
            "input_ids_student": padded_token_list_student, 
            "attention_mask_student": att_mask_student,
            "token_type_ids_student": tok_type_id_student,
            "lbl_student": onehot_encoded_lbl_student
        }
        
        return output
    
    def __len__(self):
        return len(self.df_data_teacher)


# Tokenize dataset

# In[25]:


train_data_cmp = CompDataset(df_train_t, df_train_student)
valid_data_cmp = CompDataset(df_valid_t, df_valid_student)
test_data_cmp = CompDataset(df_test_t, df_test_student)


# Create dataloader

# In[26]:


train_dataloader = DataLoader(train_data_cmp, batch_size = BATCH_SIZE)
valid_dataloader = DataLoader(valid_data_cmp, batch_size = BATCH_SIZE)
test_dataloader = DataLoader(test_data_cmp, batch_size = BATCH_SIZE)


# ## Model

# Transfer Learning model as per Bandyopadhyay, D., et al (2022) paper

# In[27]:


# bert_student_model = BertModel.from_pretrained(
#             MBERT_TYPE,
#             num_labels = 3,
#             output_hidden_states=True
#         )
# bert_student_model = bert_student_model.to(device)


# In[28]:


# optimizer_student = AdamW(
#     bert_student_model.parameters(), 
#     lr=STUDENT_LRATE
# )


# In[29]:


class TransferLearningPaper(PreTrainedModel):
    def __init__(self, config, lambda_kld, learningrate_student):
        super(TransferLearningPaper, self).__init__(config)
        
        self.bert_model_teacher = BertModel.from_pretrained(
            MODEL_TYPE, # using pretrained mBERT in INA language
            num_labels = 3,
            output_hidden_states=True
        )
    
        self.bert_model_student = BertModel.from_pretrained(
            MBERT_TYPE,
            num_labels = 3,
            output_hidden_states=True
        )
        self.optimizer_student = AdamW(
            self.bert_model_student.parameters(), 
            lr=learningrate_student
        )
        
        self.linear = nn.Linear(config.hidden_size, 3)  # Linear layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        
        # Initialize the weights of the linear layer
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        
        self.lambda_kld = lambda_kld
    
    def forward(self, input_ids_teacher, attention_mask_teacher, token_type_ids_teacher, lbl_teacher, input_ids_student, attention_mask_student, token_type_ids_student, lbl_student):
        # assume the label is already one-hot encoded
        
        self.bert_model_teacher.eval()
        self.bert_model_student.eval()
        
        with torch.no_grad():
            outputs_teacher = self.bert_model_teacher(input_ids=input_ids_teacher, attention_mask=attention_mask_teacher, token_type_ids=token_type_ids_teacher)
            outputs_student = self.bert_model_student(input_ids=input_ids_student, attention_mask=attention_mask_student, token_type_ids=token_type_ids_student)
        
            # take CLS token of the last hidden state
            pooled_output_teacher = outputs_teacher[0][:, 0, :]
            pooled_output_student = outputs_student[0][:, 0, :]
        
        linear_output = self.linear(pooled_output_student) # the output's logits
        softmax_linear_output = F.log_softmax(linear_output, dim=1)
        
        lbl_student = lbl_student[:,0,:].float()
        lbl_teacher = lbl_teacher[:,0,:].float()
        softmax_linear_output = softmax_linear_output.float()
        
        cross_entropy_loss = self.cross_entropy(softmax_linear_output, lbl_student)
        total_kld = self.kld(F.log_softmax(pooled_output_student, dim=1), F.softmax(pooled_output_teacher, dim=1))
        
        joint_loss = cross_entropy_loss + (self.lambda_kld * total_kld )
        
        return {"loss": joint_loss, "logits": softmax_linear_output}
    
    def update_param_student_model(self, loss):
        # Doing customized backpropagation for student's model
        self.optimizer_student.zero_grad()
        loss.backward()
        self.optimizer_student.step()


# In[30]:


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
    lambda_kld = 0.25, # antara 0.01-0.5
    learningrate_student = STUDENT_LRATE
)
transferlearning_model = transferlearning_model.to(device)


# ## Training

# Collect garbage

# In[31]:


gc.collect()


# Function to compute metrics

# In[32]:


def compute_metrics(p):
    print("Computing metrics...")
    pred, labels = p
    pred = np.argmax(pred[:,0,:], axis=1)
    print("pred:", pred)
    print("labels", labels)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    
    print("f1 score:", f1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


# Manual training function

# In[33]:


def train(the_model, train_data):
    the_model.train()
    
    batch_loss = 0
    
    for batch, data in enumerate(train_data):
        input_ids_teacher = data["input_ids_teacher"].to(device)
        attention_mask_teacher = data["attention_mask_teacher"].to(device)
        token_type_ids_teacher = data["token_type_ids_teacher"].to(device)
        lbl_teacher = data["lbl_teacher"].to(device)
        input_ids_student = data["input_ids_student"].to(device)
        attention_mask_student = data["attention_mask_student"].to(device)
        token_type_ids_student = data["token_type_ids_student"].to(device)
        lbl_student = data["lbl_student"].to(device)
        
        output = the_model(
            input_ids_teacher = input_ids_teacher, 
            attention_mask_teacher = attention_mask_teacher, 
            token_type_ids_teacher = token_type_ids_teacher, 
            lbl_teacher = lbl_teacher, 
            input_ids_student = input_ids_student, 
            attention_mask_student = attention_mask_student, 
            token_type_ids_student = token_type_ids_student, 
            lbl_student = lbl_student
        )
        
        loss_model = output["loss"]
        batch_loss += loss_model
        wandb.log({"loss": loss_model})
        
        # Backpropagation
        the_model.update_param_student_model(loss_model)
    
    training_loss = batch_loss / BATCH_SIZE
    
    return training_loss


# In[34]:


def validate(the_model, valid_data):
    the_model.eval()
    
    batch_loss = 0
    
    with torch.no_grad():
        for batch, data in enumerate(valid_data):
            input_ids_teacher = data["input_ids_teacher"].to(device)
            attention_mask_teacher = data["attention_mask_teacher"].to(device)
            token_type_ids_teacher = data["token_type_ids_teacher"].to(device)
            lbl_teacher = data["lbl_teacher"].to(device)
            input_ids_student = data["input_ids_student"].to(device)
            attention_mask_student = data["attention_mask_student"].to(device)
            token_type_ids_student = data["token_type_ids_student"].to(device)
            lbl_student = data["lbl_student"].to(device)

            output = the_model(
                input_ids_teacher = input_ids_teacher, 
                attention_mask_teacher = attention_mask_teacher, 
                token_type_ids_teacher = token_type_ids_teacher, 
                lbl_teacher = lbl_teacher, 
                input_ids_student = input_ids_student, 
                attention_mask_student = attention_mask_student, 
                token_type_ids_student = token_type_ids_student, 
                lbl_student = lbl_student
            )

            loss_model = output["loss"]
            batch_loss += loss_model
            wandb.log({"eval_loss": loss_model})
    
        eval_loss = batch_loss / BATCH_SIZE
    
    return eval_loss


# In[35]:


def training_sequence(the_model, train_data, valid_data, epochs):
    track_train_loss = []
    track_val_loss = []
    
    t = trange(epochs, colour="green", position=0, leave=True)
    for ep in t:
        training_loss = train(the_model, train_data)
        valid_loss = validate(the_model, valid_data)
        
        track_train_loss.append(training_loss)
        track_val_loss.append(valid_loss)
        
        t.set_description(f"Epoch [{ep + 1}/{epochs}] - Training loss: {training_loss:.2f} Validation loss: {valid_loss:.2f}")
        
        if valid_loss < min(track_val_loss) or ep + 1 == 1:
            the_model.save_pretrained(
                save_directory = MODEL_PATH + "indojavanesenli-transfer-learning"
            )
    return {
        "training_loss": track_train_loss,
        "validation_loss": track_val_loss
    }


# In[36]:


training_sequence(transferlearning_model, train_dataloader, valid_dataloader, NUM_EPOCHS)


# In[38]:


transferlearning_model.save_pretrained(save_directory = MODEL_PATH + "indojavanesenli-transfer-learning")


# In[37]:


wandb.finish()


# Training using Trainer from Huggingface (couldn't work)

# In[30]:


# training_args = TrainingArguments(
#     output_dir=MODEL_PATH + "indojavanesenli-transfer-learning/",
#     save_strategy="no", # no
#     evaluation_strategy="epoch",
#     logging_strategy="epoch",
#     learning_rate=L_RATE,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     overwrite_output_dir=True,
#     num_train_epochs=NUM_EPOCHS,
#     weight_decay=LAMBDA_L2,
#     hub_token=HF_TOKEN,
#     report_to="wandb",
#     gradient_accumulation_steps=1000,
#     push_to_hub=False,
#     run_name="transfer-learning-paper-lambda-0.25"
# )

# trainer = Trainer(
#     model=transferlearning_model.to(device),
#     args=training_args,
#     train_dataset=train_data_cmp,
#     eval_dataset=valid_data_cmp,
#     compute_metrics=compute_metrics
# )

# trainer.train()


# In[ ]:


# fin_eval = trainer.evaluate()
# wandb.log({"f1_score": fin_eval["eval_f1_score"], "eval_loss": fin_eval["eval_loss"], "accuracy": fin_eval["eval_accuracy"], "precision": fin_eval["eval_precision"], "recall": fin_eval["eval_recall"]})


# In[ ]:


# trainer.save_model()


# In[ ]:




