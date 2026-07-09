import numpy as np
import torch 
from torch import init_num_threads, nn 
from torch.utils.data import DataLoader 
import time 
import copy 
from torchattacks import PGD, FGSM
from argParse import * 
from datasetPre import * 
import pickle
from gram import select_model
from model import * 
from scipy.interpolate import CubicSpline
# from fetch_model import fetch_classifier

def eval(model, args, data_loader_test, epoch): # eval(model, args, data_loader_test, data_loader_tta, epoch)
    """ Evaluation Loop """
    model.train() # evaluation mode //////train
    device = get_device(args.g)
    model = model.to(device)
    if args.data_parallel: # use Data Parallelism with Multi-GPU 
        model = nn.DataParallel(model)
    results = [] # prediction results
    labels = [] 
    time_sum = 0.0 
    J_t = []
    for batch in data_loader_test: # Same dataset data_loader_tta 
    # for batch, batch_tta in zip(data_loader_test, data_loader_test): # Same dataset data_loader_tta 
        batch = [t.to(device) for t in batch]
        # batch_tta = [t.to(device) for t in batch_tta]
        with torch.no_grad(): # evaluation without gradient calculation 
            start_time = time.time()
            inputs, label = batch
            # inputs_tta, label_tta = batch_tta
            result,loss,gram_dist = model.forward_tta(inputs, inputs, epoch) # model inputs_tta forward_tta  inputs, inputs,epoch
            J_t.append(gram_dist)
            time_sum += time.time() - start_time
            results.append(result)
            labels.append(label)
    distance = torch.tensor(J_t)
    print('distance_sum:',torch.sum(distance))
    # print('loss:',loss)

    label = torch.cat(labels, 0)
    predict = torch.cat(results, 0)
    return stat_acc_f1(label.cpu().numpy(), predict.cpu().numpy()), torch.sum(distance)

def cotta(args):
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)
    criterion = nn.CrossEntropyLoss()
    distance = []
    
    # Build base model (encoder + classifier)
    classifier = fetch_classifier(args, input=args.encoder_cfg.hidden, output=args.activity_label_size)
    base_model = CompositeClassifier(args.encoder_cfg, classifier=classifier)
    
    device = get_device(args.g)
    
    # ------------------------------------------------------------
    # Load source model: either from --e or via select_model
    # ------------------------------------------------------------
    if args.e is not None:
        print(f"Loading encoder from {args.e}")
        checkpoint = torch.load(args.e, map_location=device)
        # Rename keys: 'transformer.' -> 'encoder.' and keep only encoder parts
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("transformer."):
                new_k = k.replace("transformer.", "encoder.", 1)
                new_state_dict[new_k] = v
            elif k.startswith("encoder."):
                new_state_dict[k] = v
            # Discard all other keys (fc, linear, decoder, norm, classifier, etc.)
        base_model.load_state_dict(new_state_dict, strict=False)
        print("Encoder loaded. GRU classifier will be adapted.")
    else:
        best_model_path = select_model(args, data_loader_train, data_loader_test)
        print("The selected model is", best_model_path)
        checkpoint = torch.load(best_model_path, map_location=device)
        base_model.load_state_dict(checkpoint)
    
    # ------------------------------------------------------------
    # Set up optimizer and adaptation model
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(params=base_model.parameters(), lr=args.lr)
    cotta_model = CoTTA_attack(model=base_model, optimizer=optimizer, arg=args)
    cotta_model = cotta_model.to(device)
    if args.data_parallel:
        cotta_model = nn.DataParallel(cotta_model)
    
    # ------------------------------------------------------------
    # Adaptation loop
    # ------------------------------------------------------------
    for e in range(args.Ada_epoch):
        (test_acc, test_f1), distance_sum = eval(cotta_model, args, data_loader_test, e)
        distance.append([0, distance_sum, test_acc, test_f1])
        for i in range(len(distance)):
            distance[i][0] = i
        distance_sort = sorted(distance, key=lambda x: x[1], reverse=True)
        print("Distance_Sort", distance_sort)
        print('Epoch %d/%d , Accuracy: %0.3f, F1: %0.3f'
              % (e+1, args.Ada_epoch, test_acc, test_f1))
        if e == 50:
            print('The Total Epoch have been reached.')
if __name__ == "__main__": 
    args = set_arg()
    set_seeds(args.seed)
    print("Seed number in Adapt:", args.seed)
    cotta(args)