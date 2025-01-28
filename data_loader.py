import torch
from PIL import Image
from torch.utils.data import Dataset
import re

def text_preprocessing(text):

    text = re.sub(r'(@.*?)[\s]', ' ', text)

    text = re.sub(r'&amp;', '&', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

class FakeNewsDataset(Dataset):

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]
    
    def pre_processing_BERT(self, sent):

        input_ids = []
        attention_mask = []
        
        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  
            add_special_tokens=True,        
            max_length=self.MAX_LEN,        
            padding='max_length',           
            # return_tensors='pt',         
            return_attention_mask=True,     
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.root_dir + self.csv_data['image_id'][idx] + '.jpg'
        image = Image.open(img_name).convert("RGB")
        image = self.image_transform(image)
        
        text = self.csv_data['post_text'][idx]
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]

        if label == 'fake':
            label = '1'
        else:
            label = '0'
        label = int(label)
        
        label = torch.tensor(label)

        sample = {
                  'image_id'  :  image, 
                  'BERT_ip'   : [tensor_input_id, tensor_input_mask],
                  'label'     :  label
                  }

        return sample