import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)

        #for name, param in self.clip.named_parameters():
        # top layers always need to train
         #   if 'ln_final.weight' in name or 'ln_final.bias' in name or 'token_embedding.weight' in name or 'text_projection' in name \
          #          or 'visual.ln_post.weight' in name or 'visual.ln_post.bias' in name or 'positional_embedding' in name or 'visual_proj' in name \
           #         or 'logit_scale' in name:
            #    continue # need to train

            #if 'transformer.gamma' in name or 'transformer.beta' in name:
               # print('true true........')
              #  continue

            #param.requires_grad = False


       # self.adapter = torch.nn.DataParallel(nn.Linear(feat_dim, feat_dim, bias=False)).cuda()


       # self.init_models()

    def init_models(self, optimizer=True):
        self.model_optim_params_list = []

        self.freeze_layers_clip(self.clip, 12)

        print("Using", torch.cuda.device_count(), "GPUs.")

        self.visual_model = torch.nn.DataParallel(self.clip.vision_model).cuda()

        self.text_model = torch.nn.DataParallel(self.clip.text_model).cuda()

        self.adapter = torch.nn.DataParallel(nn.Linear(512, 512, bias=False)).cuda()

        self.model_optim_params_list.append({'params': self.adapter.parameters(),
                                             'lr': 0.2,
                                             'momentum': 0.9,
                                             'weight_decay': 0.0005})

        self.model_optim_params_list.append({'params': self.visual_model.parameters(),
                                             'lr': 0.00001,
                                             'momentum': 0.9,
                                             'weight_decay': 0.005})

        self.model_optim_params_list.append({'params': self.text_model.parameters(),
                                             'lr': 0.00001,
                                             'momentum': 0.9,
                                             'weight_decay': 0.0005})

    def freeze_layers_clip(model, freeze_layer_num):
        assert hasattr(model, 'clip')
        assert freeze_layer_num <= 12 and freeze_layer_num >= -1

        if freeze_layer_num == -1:
            return

        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if 'final_layer_norm' in name or 'text_projection' in name \
                    or 'post_layernorm' in name or 'visual_projection' in name \
                    or 'logit_scale' in name:
                continue  # need to train

            elif 'text_model.encoder.layers' in name or 'vision_model.encoder.layers' in name:
                layer_num = int(name.split('.layers.')[1].split('.')[0])
                if layer_num >= freeze_layer_num:
                    continue  # need to train

            print(name)
            param.requires_grad = False

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)

        #x = self.adapter(image_features)
        #ratio = 0.2
        #x = ratio * x + (1 - ratio) * image_features
   
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled

