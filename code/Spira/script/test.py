import os
import math
import torch
import torch.nn as nn
import traceback
import pandas as pd

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

from models.spiraconv import *

from models.panns import *

from utils.audio_processor import AudioProcessor 

from sklearn.metrics import f1_score, recall_score

from utils.models import return_model


import random
# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
def test(criterion, ap, model, c, testloader, step,  cuda, confusion_matrix=False, debug=False):
    losses = []
    accs = []
    model.zero_grad()
    model.eval()
    loss = 0
    loss_control = 0
    loss_patient = 0 
    acc = 0
    preds = []
    targets = []
    names = []
    with torch.no_grad():
        for feature, target, slices, targets_org, name in testloader:       
            #try:
            if cuda:
                feature = feature.cuda()
                target = target.cuda()
            output = model(feature).float()

            # output = torch.round(output * 10**4) / (10**4)

            # Calculate loss
            if c.dataset["temporal_control"] == "overlapping":
                # unpack overlapping for calculation loss and accuracy 
                if slices is not None and targets_org is not None:
                    idx = 0
                    new_output = []
                    new_target = []
                    for i in range(slices.size(0)):
                        num_samples = int(slices[i].cpu().numpy())

                        samples_output = output[idx:idx+num_samples]
                        output_mean = samples_output.mean()
                        samples_target = target[idx:idx+num_samples]
                        target_mean = samples_target.mean()

                        new_target.append(target_mean)
                        new_output.append(output_mean)
                        idx += num_samples

                    target = torch.stack(new_target, dim=0)
                    output = torch.stack(new_output, dim=0)
                    #print(target, targets_org)
                    if cuda:
                        output = output.cuda()
                        target = target.cuda()
                        targets_org = targets_org.cuda()
                    if not torch.equal(targets_org, target):
                        raise RuntimeError("Integrity problem during the unpack of the overlay for the calculation of accuracy and loss. Check the dataloader !!")

            loss += criterion(output, target).item()
            idxs = (target == c.dataset['control_class'])
            loss_control += criterion(output[idxs], target[idxs]).item()
            idxs = (target == c.dataset['patient_class'])
            loss_patient += criterion(output[idxs], target[idxs]).item()
            

            # calculate binnary accuracy
            y_pred_tag = torch.round(output)
            acc += (y_pred_tag == target).float().sum().item()
            preds += y_pred_tag.reshape(-1).int().cpu().numpy().tolist()
            targets += target.reshape(-1).int().cpu().numpy().tolist()
            names += name

        targets = np.array(targets)
        preds = np.array(preds)
        names = np.array(names)
        idxs = np.nonzero(targets == c.dataset['control_class'])
        control_target = targets[idxs]
        control_preds = preds[idxs]
        names_control = names[idxs]

        idxs = np.nonzero(targets == c.dataset['patient_class'])
        
        patient_target = targets[idxs]
        patient_preds = preds[idxs]
        names_patient = names[idxs]

        if debug:
            print('+'*40)
            print("Control Files Classified incorrectly:")
            incorrect_ids = np.nonzero(control_preds != c.dataset['control_class'])
            inc_names = names_control[incorrect_ids]
            print("Num. Files:", len(inc_names))
            print(inc_names)
            print('+'*40)
            print('-'*40)
            print("Patient Files Classified incorrectly:")
            incorrect_ids = np.nonzero(patient_preds != c.dataset['patient_class'])
            inc_names = names_patient[incorrect_ids]
            print("Num. Files:", len(inc_names))
            print(inc_names)
            print('-'*40)


        acc_control = (control_preds == control_target).mean()
        acc_patient = (patient_preds == patient_target).mean()
        acc_balanced = (acc_control + acc_patient) / 2

        loss_control = loss_control / len(control_target)
        loss_patient = loss_patient / len(patient_target)

        loss_balanced = (loss_control + loss_patient) / 2
        f1 = f1_score(targets.tolist(), preds.tolist())
        # print(targets.shape, preds.shape)
        uar = recall_score(targets.tolist(), preds.tolist(), average='macro')
        if confusion_matrix:
            print("======== Confusion Matrix ==========")
            y_target = pd.Series(targets, name='Target')
            y_pred = pd.Series(preds, name='Predicted')
            df_confusion = pd.crosstab(y_target, y_pred, rownames=['Target'], colnames=['Predicted'], margins=True)
            print(df_confusion)
            
        mean_acc = acc / len(testloader.dataset)
        mean_loss = loss / len(testloader.dataset)
    print("Test\n ", "Acurracy: ", mean_acc, "Acurracy Control: ", acc_control, "Acurracy Patient: ", acc_patient, "Acurracy Balanced", acc_balanced)
    print("Loss:", mean_loss, "Loss Control:", loss_control, "Loss Patient:", loss_patient, "Loss balanced: ", loss_balanced)
    print("F1:", f1, "UAR:", uar)
    return mean_acc


def run_test(args, checkpoint_path, testloader, c, model_name, ap, cuda=True, debug=False):

    # define loss function
    criterion = nn.BCELoss(reduction='sum')
    model = return_model(c)

    step = 0
    if checkpoint_path is not None:
        print("Loading checkpoint: %s" % checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("Model Sucessful Load !")
        except Exception as e:
            raise ValueError("You need pass a valid checkpoint, may be you need check your config.json because de the of this checkpoint cause the error: "+ e)       
        step = checkpoint['step']
    else:
        raise ValueError("You need pass a checkpoint_path")   

    # convert model from cuda
    if cuda:
        model = model.cuda()

    model.train(False)
    test_acc = test(criterion, ap, model, c, testloader, step, cuda=cuda, confusion_matrix=True, debug=debug)
        

if __name__ == '__main__':
    # python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/checkpoint_1068.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/config.json  --batch_size 5 --num_workers 2 --no_insert_noise True

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_csv', type=str, required=True,
                        help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv")
    parser.add_argument('-r', '--test_root_dir', type=str, required=True,
                        help="Test root dir example: ../SPIRA_Dataset_V1/")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations get in checkpoint path")
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size for test")
    parser.add_argument('--num_workers', type=int, default=10,
                        help="Number of Workers for test data load")
    parser.add_argument('--no_insert_noise', type=bool, default=True,
                        help=" No insert noise in test ?")
    parser.add_argument('--num_noise_control', type=int, default=1,
                        help="Number of Noise for insert in control")
    parser.add_argument('--num_noise_patient', type=int, default=0,
                        help="Number of Noise for insert in patient")
    parser.add_argument('--debug', type=bool, default=True,
                        help=" Print classification error")
                        
    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(**c.audio)
    
    if not args.no_insert_noise:
        c.data_aumentation['insert_noise'] = True
    else:
        c.data_aumentation['insert_noise'] = False

    # ste values for noisy insertion in test
    c.data_aumentation["num_noise_control"] = args.num_noise_control
    c.data_aumentation["num_noise_patient"] = args.num_noise_patient

    print("Insert noise ?", c.data_aumentation['insert_noise'])

    c.dataset['test_csv'] = args.test_csv
    c.dataset['test_data_root_path'] = args.test_root_dir


    c.test_config['batch_size'] = args.batch_size
    c.test_config['num_workers'] = args.num_workers
    max_seq_len = c.dataset['max_seq_len'] 

    test_dataloader = test_dataloader(c, ap, max_seq_len=max_seq_len)

    run_test(args, args.checkpoint_path, test_dataloader, c, c.model_name, ap, cuda=True, debug=args.debug)