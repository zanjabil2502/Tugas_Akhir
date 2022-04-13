# Reference: The models here is adapted from: https://github.com/qiuqiangkong/audioset_tagging_cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from utils.panns import *

from utils.panns import _ResNet, _ResnetBasicBlock

from utils.generic_utils import SpecAugmentation, do_mixup, set_init_dict

class Cnn14(nn.Module):
    def __init__(self, classes_num, spec_aug=True):
        
        super(Cnn14, self).__init__()
        top_db = None
        self.spec_aug = spec_aug
        # torchaudio generate mel spectogram without logmel so we need aplied log mel because panns is trained wit this
        self.amplitude_to_DB =  torchaudio.transforms.AmplitudeToDB('power', top_db)

        # Spec augmenter
        if self.spec_aug:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = input.unsqueeze(1) # (batch_size, 1, time_steps, mel_bins)
        x = self.amplitude_to_DB(x)   
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and self.spec_aug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(self, c):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        classes_num = c.model['num_class']
        freeze_base = c.model['freeze_base']
        spec_aug = True if 'spec_aug' not in c.model.keys() else c.model['spec_aug']
        self.base = Cnn14(audioset_classes_num, spec_aug)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

        if c.model['pretreined_checkpoint'] and self.training:
            self.load_from_pretrain(c.model['pretreined_checkpoint'])

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        try:
            self.base.load_state_dict(checkpoint['model'])
            print(" Pretrained checkpoint loaded: ", pretrained_checkpoint_path)
        except:
            print(" > Partial model initialization.")
            model_dict = self.base.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint)
            self.base.load_state_dict(model_dict)
            print("Partial Pretrained checkpoint loaded: ", pretrained_checkpoint_path)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        x = self.fc_transfer(embedding)
        x = torch.sigmoid(x) 
        return x

class ResNet38(nn.Module):
    def __init__(self, classes_num, spec_aug=True):
        super(ResNet38, self).__init__()
        top_db = None
        self.spec_aug = spec_aug
        # torchaudio generate mel spectogram without logmel so we need aplied log mel because panns is trained wit this
        self.amplitude_to_DB =  torchaudio.transforms.AmplitudeToDB('power', top_db)

        # Spec augmenter
        if self.spec_aug:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        
        x = input.unsqueeze(1) # (batch_size, 1, time_steps, mel_bins)
        x = self.amplitude_to_DB(x)   
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and self.spec_aug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Transfer_ResNet38(nn.Module):
    def __init__(self, c):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet38, self).__init__()
        audioset_classes_num = 527
        
        classes_num = c.model['num_class']
        freeze_base = c.model['freeze_base']
        spec_aug = True if 'spec_aug' not in c.model.keys() else c.model['spec_aug']

        self.base = ResNet38(audioset_classes_num, spec_aug)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

        if c.model['pretreined_checkpoint'] and self.training:
            self.load_from_pretrain(c.model['pretreined_checkpoint'])

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        try:
            self.base.load_state_dict(checkpoint['model'])
            print(" Pretrained checkpoint loaded: ", pretrained_checkpoint_path)
        except:
            print(" > Partial model initialization.")
            model_dict = self.base.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint)
            self.base.load_state_dict(model_dict)
            print("Partial Pretrained checkpoint loaded: ", pretrained_checkpoint_path)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        x = self.fc_transfer(embedding)
        x = torch.sigmoid(x) 
        return x


class MobileNetV1(nn.Module):
    def __init__(self, classes_num, spec_aug=True):
        
        super(MobileNetV1, self).__init__()

        top_db = None
        self.spec_aug = spec_aug
        # torchaudio generate mel spectogram without logmel so we need aplied log mel because panns is trained wit this
        self.amplitude_to_DB =  torchaudio.transforms.AmplitudeToDB('power', top_db)

        # Spec augmenter
        if self.spec_aug:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(inp), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.features = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = input.unsqueeze(1) # (batch_size, 1, time_steps, mel_bins)

        x = self.amplitude_to_DB(x)   
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and self.spec_aug:
            x = self.spec_augmenter(x)
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.features(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Transfer_MobileNetV1(nn.Module):
    def __init__(self, c):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_MobileNetV1, self).__init__()
        audioset_classes_num = 527
        
        classes_num = c.model['num_class']
        freeze_base = c.model['freeze_base']
        spec_aug = True if 'spec_aug' not in c.model.keys() else c.model['spec_aug']

        self.base = MobileNetV1(audioset_classes_num, spec_aug)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

        if c.model['pretreined_checkpoint'] and self.training:
            self.load_from_pretrain(c.model['pretreined_checkpoint'])

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        try:
            self.base.load_state_dict(checkpoint['model'])
            print(" Pretrained checkpoint loaded: ", pretrained_checkpoint_path)
        except:
            print(" > Partial model initialization.")
            model_dict = self.base.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint)
            self.base.load_state_dict(model_dict)
            print("Partial Pretrained checkpoint loaded: ", pretrained_checkpoint_path)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        x = self.fc_transfer(embedding)
        x = torch.sigmoid(x) 
        return x