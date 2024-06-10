# using mmpretrain module

import torch
from mmpretrain import get_model, inference_model
import time
def main():
    model = get_model('riformer-s12_in1k', pretrained=True)

    batch_size_list = [1,128]
        
    model = model.cuda()
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
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
        print(f"batch size {batch_size}' average run time : {(end_time - start_time)/avg_time}")
        print("=========================================================")

    print("repparameterized model")
    #model = repvgg_model_convert(model, save_path=None, do_copy=True)
    model = model.cuda()
    model.backbone.switch_to_deploy()
    for batch_size in batch_size_list:
        gen_img = torch.randn(batch_size, 3, 224, 224)
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
        print(f"batch size {batch_size}' average run time : {(end_time - start_time)/avg_time}")
        print("=========================================================")

if __name__ == '__main__':
    main()
    print("done")