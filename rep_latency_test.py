import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint
import time
import pandas as pd
import timm
from timm.utils.model import reparameterize_model
import os
from easydict import EasyDict as edict
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='repvgg_a0')
    return parser.parse_args()

TIMM_SUPPORT_REPARAM = ['repvgg', 'mobileone', 'repvit', 'levit', 'efficientvit', 'repghost', 'fastvit']

if __name__ == '__main__':
    
    args = get_args()
    
    reparam_trigger = False
    for model_name in TIMM_SUPPORT_REPARAM:
        if model_name in args.model.lower():
            reparam_trigger = True
            break
    if reparam_trigger == False:
        print(f"model {args.model} does not support reparameterization")
        exit()

    model = timm.create_model(args.model, pretrained=True)
    model.eval()
    df = pd.DataFrame()
    run_config = edict()
    os_info = os.uname().machine
    
    if os_info == 'x86_64':
        print("server mode")
        run_config.device = ['cpu']
        run_config.batch_size = [1, 128]
    elif os_info == 'aarch64':
        print("edge mode")
        run_config.device = ['cpu']
        run_config.batch_size = [1]
    
    if torch.cuda.is_available():
        run_config.device.append('gpu')
        
    
    for os_mode in run_config.device:
        model = timm.create_model(args.model, pretrained=True)
        model.eval()
        for batch_size in run_config.batch_size:
            gen_img = torch.randn(batch_size, 3, 224, 224)
            if os_mode == 'gpu':
                gen_img = gen_img.cuda()
                model = model.cuda()
            print(f"warmup time ({os_mode}): batch_size: {batch_size}")
            for _ in range(5):
                out=model(gen_img)
            
            print(f"start: batch size {batch_size}")

            start_time = time.time()
            avg_time = 10
            for _ in range(avg_time):
                out = model(gen_img)

            end_time = time.time()
            print(f"batch size {batch_size}' average run time ({os_mode}): {(end_time - start_time)/avg_time}")
            result_df = pd.DataFrame({"type": os_mode, "batch_size": batch_size, "average_run_time": (end_time - start_time)/avg_time, "Reparam": 'No'}, index=[0])
            df = pd.concat([df, result_df])
            print("=========================================================")

        print("reparameterized model")
        model = reparameterize_model(model)
        if os_mode == 'gpu':
            model = model.cuda()
        for batch_size in run_config.batch_size:
            gen_img = torch.randn(batch_size, 3, 224, 224)
            if os_mode == 'gpu':
                gen_img = gen_img.cuda()

            print(f"warmup time: batch_size: {batch_size}")
            for _ in range(5):
                out=model(gen_img)
            
            print(f"start: batch size {batch_size}")

            start_time = time.time()
            avg_time = 10
            for _ in range(avg_time):
                out = model(gen_img)

            end_time = time.time()
            print(f"batch size {batch_size}' average run time ({os_mode}): {(end_time - start_time)/avg_time}")
            result_df = pd.DataFrame({"type": os_mode, "batch_size": batch_size, "average_run_time": (end_time - start_time)/avg_time, "Reparam": 'Yes'}, index=[0])
            df = pd.concat([df, result_df])
            print("=========================================================")
    print(df)
    df.to_csv(f"latency_{args.model}.csv")
