# using mmpretrain module

import torch
from mmpretrain import get_model, inference_model
import time
import pandas as pd

def main():
    model = get_model('riformer-s12_in1k', pretrained=True)

    batch_size_list = [1,128]
    print("**"*25)
    print("run cpu mode")
    model = model
    df = pd.DataFrame()
    
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
        gen_img = gen_img

        print(f"warmup time (cpu): batch_size: {batch_size}")
        for _ in range(5):
            out=model(gen_img)
        
        print(f"start: batch size {batch_size}")

        start_time = time.time()
        avg_time = 10
        for _ in range(avg_time):
            out = model(gen_img)

        end_time = time.time()
        print(f"batch size {batch_size}' average run time (cpu) : {(end_time - start_time)/avg_time}")
        result_df = pd.DataFrame({"type":"cpu", "batch_size": batch_size, "time": (end_time - start_time)/avg_time, "Reparam": 'No'}, index=[0])
        df = pd.concat([df, result_df], ignore_index=True)

        print("=========================================================")

    print("repparameterized model")
    model.backbone.switch_to_deploy()
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
        gen_img = gen_img

        print(f"warmup time (cpu): batch_size: {batch_size}")
        for _ in range(5):
            out=model(gen_img)
        
        print(f"start: batch size {batch_size}")

        start_time = time.time()
        avg_time = 10
        for _ in range(avg_time):
            out = model(gen_img)

        end_time = time.time()
        print(f"batch size {batch_size}' average run time (cpu) : {(end_time - start_time)/avg_time}")
        result_df = pd.DataFrame({"type":"cpu", "batch_size": batch_size, "time": (end_time - start_time)/avg_time, "Reparam": 'Yes'}, index=[0])
        df = pd.concat([df, result_df], ignore_index=True)
        print("=========================================================")

    model = get_model('riformer-s12_in1k', pretrained=True)

    batch_size_list = [1,128]
    print("**"*25)
    print("run gpu mode")
    model = model.cuda()
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
        gen_img = gen_img.cuda()

        print(f"warmup time (gpu): batch_size: {batch_size}")
        for _ in range(5):
            out=model(gen_img)
        
        print(f"start: batch size {batch_size}")

        start_time = time.time()
        avg_time = 10
        for _ in range(avg_time):
            out = model(gen_img)

        end_time = time.time()
        print(f"batch size {batch_size}' average run time (gpu): {(end_time - start_time)/avg_time}")
        result_df = pd.DataFrame({"type":"gpu", "batch_size": batch_size, "time": (end_time - start_time)/avg_time, "Reparam": 'No'}, index=[0])
        df = pd.concat([df, result_df], ignore_index=True)

        print("=========================================================")

    print("repparameterized model")
    #model = repvgg_model_convert(model, save_path=None, do_copy=True)
    model = model.cuda()
    model.backbone.switch_to_deploy()
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
        gen_img = gen_img.cuda()

        print(f"warmup time (gpu): batch_size: {batch_size}")
        for _ in range(5):
            out=model(gen_img)
        
        print(f"start: batch size {batch_size}")

        start_time = time.time()
        avg_time = 10
        for _ in range(avg_time):
            out = model(gen_img)

        end_time = time.time()
        print(f"batch size {batch_size}' average run time (gpu): {(end_time - start_time)/avg_time}")
        result_df = pd.DataFrame({"type":"gpu", "batch_size": batch_size, "time": (end_time - start_time)/avg_time, "Reparam": 'Yes'}, index=[0])
        df = pd.concat([df, result_df], ignore_index=True)
        print("=========================================================")
    df.to_csv("riformer.csv", index=False)
if __name__ == '__main__':
    main()
    print("done")