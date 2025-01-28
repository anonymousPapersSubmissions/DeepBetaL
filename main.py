import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch.special as sp

from models import *
from data_loader import *
from train_val import *

reshuffle_data = True
if reshuffle_data:
    df_train = pd.read_csv("twitter/train_posts_clean.csv")
    df_test = pd.read_csv("twitter/test_posts.csv")
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    oversample = RandomOverSampler(sampling_strategy='minority')
    y = df["label"]
    X = df.drop(["label"], axis=1)
    df_marged, y_over = oversample.fit_resample(X, y)
    df_marged['label'] = y_over
    df_minority = df_marged[df_marged['label']=='real']
    print(f"df_minority: {len(df_minority)}")
    df_majority = df_marged[df_marged['label']=='fake']
    print(f"df_majority: {len(df_majority)}")
    print(f"df_marged: {len(df_marged)}")
    df_train_s, df_test_s = train_test_split(df_marged, test_size=0.2, shuffle=True, random_state=0)
    df_train_s.to_csv('twitter/df_train2.csv', encoding='utf-8', index=False)
    df_test_s.to_csv('twitter/df_test2.csv', encoding='utf-8', index=False)
    df_train = pd.read_csv("twitter/df_train2.csv")
    df_test = pd.read_csv("twitter/df_test2.csv")

    print(f"length of training set: {len(df_train)}")
    print(f"length of test set: {len(df_test)}")

else:
    df_train = pd.read_csv("twitter/train_posts_clean.csv")
    df_test = pd.read_csv("twitter/test_posts.csv")
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    df_minority = df[df['label']=='real']
    print(f"df_minority: {len(df_minority)}")
    df_majority = df[df['label']=='fake']
    print(f"df_majority: {len(df_majority)}")

    df_train_s, df_test_s = train_test_split(df, test_size=0.2, shuffle=True, random_state=0)
    df_train_s.to_csv('twitter/df_train3.csv', encoding='utf-8', index=False)
    df_test_s.to_csv('twitter/df_test3.csv', encoding='utf-8', index=False)
    df_train = pd.read_csv("twitter/df_train3.csv")
    df_test = pd.read_csv("twitter/df_test3.csv")
    print(f"length of training set: {len(df_train)}")
    print(f"length of test set: {len(df_test)}")


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

MAX_LEN = 500
root_dir = "twitter/"

transformed_dataset_train = FakeNewsDataset(df_train, root_dir+"images_train/", image_transform, tokenizer, MAX_LEN)

transformed_dataset_val = FakeNewsDataset(df_test, root_dir+"images_test/", image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


cls_BCE = nn.BCELoss()


def kl_divergence_betaliouville(alpha, beta, beta_p, alpha_p):
    """
    Compute the KL divergence between two Beta-Liouville distributions.

    Args:
        alpha: Tensor of shape (batch_size, feature_dim), parameters of P.
        beta: Tensor of shape (batch_size, 1), beta parameter of P.
        alpha_p: Tensor of shape (batch_size, feature_dim), parameters of Q.
        beta_p: Tensor of shape (batch_size, 1), beta parameter of Q.

    Returns:
        Tensor of shape (batch_size,), the KL divergence for each batch element.
    """
    alpha = torch.max(torch.tensor(0.0001), alpha).to(device)
    alpha_p = alpha_p.expand_as(alpha)
    # Compute terms related to normalization constants
    term_const = (
        sp.gammaln(alpha.sum(dim=-1)) - sp.gammaln(alpha_p.sum(dim=-1))
        + sp.gammaln(alpha.sum(dim=-1) + beta.squeeze(-1)) - sp.gammaln(alpha_p.sum(dim=-1) + beta_p.squeeze(-1))
        - (sp.gammaln(alpha).sum(dim=-1) - sp.gammaln(alpha_p).sum(dim=-1))
        - (sp.gammaln(beta.squeeze(-1)) - sp.gammaln(beta_p.squeeze(-1)))
    )

    # Compute expectation terms
    psi_alpha = sp.digamma(alpha)
    psi_alpha_sum_beta = sp.digamma(alpha.sum(dim=-1, keepdim=True) + beta)

    term_alpha = ((alpha - alpha_p) * (psi_alpha - psi_alpha_sum_beta)).sum(dim=-1)

    psi_alpha_total = sp.digamma(alpha.sum(dim=-1, keepdim=True))
    psi_beta_total = sp.digamma(beta)
    psi_alpha_beta = sp.digamma(alpha.sum(dim=-1, keepdim=True) + beta)

    term_alpha_beta = (alpha.sum(dim=-1) - alpha_p.sum(dim=-1)) * (
        psi_alpha_total.squeeze(-1) - psi_alpha_beta.squeeze(-1)
    )

    term_beta = (beta.squeeze(-1) - beta_p.squeeze(-1)) * (
        psi_beta_total.squeeze(-1) - psi_alpha_beta.squeeze(-1)
    )

    # Combine terms to get KL divergence
    kl = term_const + term_alpha + term_alpha_beta + term_beta

    return torch.mean(kl)



def loss_fn(logits, b_labels, alpha_smoothed):

    cls_loss = cls_BCE(logits, b_labels)
    
    beta = torch.tensor(1.5, device=alpha_smoothed.device)  # Posterior Beta
    beta_p = torch.tensor(1.0, device=alpha_smoothed.device)  # Prior Beta
    kld_loss =  kl_divergence_betaliouville(alpha_smoothed, beta, beta_p, alpha_p=torch.tensor(0.01))
    loss = cls_loss + kld_loss * 0.01
    return loss, cls_loss

def set_seed(seed_value=42):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

parameter_dict_model={
    'text_fc2_out': 32,
    'text_fc1_out': 2742,
    'dropout_p': 0.4,
    'fine_tune_text_module': False,
    'img_fc1_out': 2742,
    'img_fc2_out': 32,
    'dropout_p': 0.4,
    'fine_tune_vis_module': False,
    'fusion_output_size': 35
}

parameter_dict_opt={'l_r': 3e-5,
                    'eps': 1e-8
                    }


EPOCHS = 20  

set_seed(7)

final_model = Text_Concat_Vision(parameter_dict_model)

final_model = final_model.to(device)

optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # 默认值
                                            num_training_steps=total_steps)

writer = SummaryWriter('multi_att_exp3')

train(model=final_model,
      loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
      train_dataloader=train_dataloader, val_dataloader=val_dataloader,
      epochs=EPOCHS, evaluation=True, #epochs=150
      device=device,
      param_dict_model=parameter_dict_model, param_dict_opt=parameter_dict_opt,
      save_best=True,
      file_path='saved_models/best_model.pt'
      , writer=writer
      )