import torch
from torchvision import models
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

device = torch.device("cuda")

class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=32, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()

        self.fine_tune_module = fine_tune_module

        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True,
                    return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()

    def forward(self, input_ids, attention_mask):

        # BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(out['pooler_output'].shape)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        )

        return x

    def fine_tune(self):

        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module

class VisionEncoder(nn.Module):

    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        super(VisionEncoder, self).__init__()

        self.fine_tune_module = fine_tune_module

        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])

        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()

    def forward(self, images):

        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        return x

    def fine_tune(self):

        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module

#LanguageAndVision
class Text_Concat_Vision(torch.nn.Module):

    def __init__(self,
        model_params
    ):
        super(Text_Concat_Vision, self).__init__()

        self.text_encoder = TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'], model_params['dropout_p'], model_params['fine_tune_text_module'])
        self.vision_encode = VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'], model_params['dropout_p'], model_params['fine_tune_vis_module'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out']),
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

        self.alpha = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )

        self.mu = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )

        self.logvar = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )

    
    def beta_liouville_covariance(self, text_features, image_features):
        """
        Compute the normalized Beta-Liouville covariance between text_features and image_features.

        Parameters:
            text_features (torch.Tensor): Text feature embeddings (batch_size x feature_dim).
            image_features (torch.Tensor): Image feature embeddings (batch_size x feature_dim).

        Returns:
            torch.Tensor: Covariance values for each batch instance (batch_size x feature_dim).
        """
        # Avoid division by zero in norm calculations by adding a small epsilon
        epsilon = 1e-8

        # Compute L2 norm for each feature set
        alpha_1 = torch.norm(text_features, p=2, dim=1, keepdim=True) + epsilon  # Shape: (batch_size x 1)
        alpha_2 = torch.norm(image_features, p=2, dim=1, keepdim=True) + epsilon  # Shape: (batch_size x 1)

        # Normalize alpha values for each dimension
        alpha_1 = text_features / alpha_1  # Shape: (batch_size x feature_dim)
        alpha_2 = image_features / alpha_2  # Shape: (batch_size x feature_dim)

        # Compute the sum of alpha over the feature dimension
        alpha_sum = alpha_1 + alpha_2 + epsilon  # Add epsilon to avoid division by zero

        # Compute intermediate terms
        term1 = (alpha_1 + 1) / (alpha_sum + 1)
        term2 = alpha_1 / alpha_sum

        factor1 = term1 * term2 / (alpha_sum + 1)
        factor2 = term2 / alpha_sum

        # Compute the covariance
        covariance = (alpha_1 * alpha_2) / alpha_sum
        covariance *= factor1 - factor2

        # Handle NaN values by replacing them with zeros
        covariance = torch.nan_to_num(covariance, nan=0.0)

        return covariance 


    def reparameterize(self, alpha, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        alpha_smoothed = eps * std + mu + alpha
        #Laplace
        rho =  F.softmax(alpha_smoothed, dim=1)
        z = torch.zeros_like(rho)
        #Transform to Beta-Liouville
        cumulative_sum = torch.zeros(rho.size(0), 1, device=rho.device)
        for i in range(rho.size(1)):
            z[:, i] = rho[:, i] * (1 - cumulative_sum.squeeze(-1))
            cumulative_sum += z[:, i].unsqueeze(-1)

        return z, alpha_smoothed


    #def forward(self, text, image, label=None):
    def forward(self, text, image, label=None):

        ## text to Bert
        text_features = self.text_encoder(text[0], text[1])
        ## image to vgg
        image_features = self.vision_encode(image)

        covariance = self.beta_liouville_covariance(text_features, image_features)

        #combined_features = torch.cat(
        #    [text_features+covariance, image_features+covariance], dim = 1
        #)

        combined_features = self.dropout(covariance)

        fused_ = self.dropout(
            torch.relu(
            self.fusion(combined_features)
            )
        )

        alpha = self.alpha(fused_)
        mu = self.mu(fused_)
        logvar = self.logvar(fused_)

        fused, alpha_smoothed = self.reparameterize(alpha, mu, logvar)
       
        prediction = torch.sigmoid(self.fc(fused))

        prediction = prediction.squeeze(-1)

        prediction = prediction.float()

        return prediction, covariance