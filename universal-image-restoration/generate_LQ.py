import numpy as np
import cv2
import os
import sys
import json
import random
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.deg_util import degrade, match_dim, random_degrade
import argparse

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# deg_type: noisy, jpeg
# param: 50 for noise_level, 10 for jpeg compression quality
def generate_LQ(source_dir=[], deg_type='blur', param=[10, 15], save_path= "./dataset/", epochs= 2, mode= "train"):
    
    print("#"*20)
    print(f"\nSource DIR: {source_dir}\nMode: {mode}\nDegradation type: {deg_type}\nParam range: {param}\nEpochs: {epochs}\nSave Path: {save_path}/{mode}/{deg_type}\n")
    print("#"*20)
    # set data dir

    for p in range(10,81,10):
        savedir_GT = f"{save_path}/{mode}/{deg_type}{p}/GT"
        savedir_LQ = f"{save_path}/{mode}/{deg_type}{p}/LQ"
        
        os.makedirs(savedir_GT, exist_ok= True)
        os.makedirs(savedir_LQ, exist_ok= True)

        filepaths = [os.path.join(source, f) for source in source_dir for f in os.listdir(source) if is_image_file(f)  ]
        num_files = len(filepaths)

        degraded_prompts = {}

        
        for epoch in range(epochs):
            # prepare data with augementation
            progress_bar = tqdm(range(num_files), dynamic_ncols=True)
            for i in progress_bar:
                filename = filepaths[i]
                progress_bar.set_description("[Epochs {}/{}] Processing -- {}".format(epoch + 1, epochs, filename.split("/")[-1]))
                # read image
                image = cv2.imread(filename)
                # crop to 512x512
                image_GT = match_dim(image, (512, 512), "random")
                # random param
                if deg_type == "blur":
                    # rand_num = random.randint(param[0], param[1])
                    rand_num = p / 10
                    rand_num = rand_num + 1 if rand_num % 2 == 0 else rand_num
                elif deg_type == "resize":
                    rand_num = round(random.uniform(param[0], param[1]), 1)
                    rand_num = float(p * 0.1)
                elif deg_type in ["noisy", "jpeg"]:
                    rand_num = random.randint(param[0], param[1])
                    rand_num = p

                # degraded it
                if deg_type != "random":
                    image_LQ = (degrade(image_GT / 255., deg_type, rand_num) * 255).astype(np.uint8)
                    # save 
                    degraded_prompt = f"{deg_type} with parameter {rand_num}"
                    # degraded_prompt = f"an image with {deg_type}"
                else :
                    deg_list = set()
                    image_LQ = (random_degrade(image_GT / 255., deg_list= deg_list) * 255).astype(np.uint8)
                    deg_text = ", ".join(deg_list)
                    degraded_prompt = f"an image with {deg_text}"

                epoch_filename = str(epoch) + "_" + filename.split("/")[-1]
                degraded_prompts["./" + epoch_filename] = degraded_prompt
                cv2.imwrite(os.path.join(savedir_GT, epoch_filename), image_GT)
                cv2.imwrite(os.path.join(savedir_LQ, epoch_filename), image_LQ)


        json_path = os.path.join(f"{save_path}/{mode}/{deg_type}{p}/", "degraded_prompts.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(degraded_prompts, f, indent=4)
        print('Finished!!!')

def setting_param():
    param = dict()
    param["noisy"] = [5, 40]
    param["resize"] = [0.5, 4]
    param["blur"] = [5, 40]
    param["jpeg"] = [30, 92]
    param["random"] = None
    # 2,41,5: blur sigma x
    return param

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default= ["/mnt/hdd7/yicheng/daclip-uir/datasets/lsdir"], type= list)
    parser.add_argument('--save_path', default= "./datasets_1/train", type= str)
    parser.add_argument('--deg_type', choices= ["noisy", "resize", "blur", "jpeg", "random"], default= "random", type= str)
    parser.add_argument('--mode', choices= ["train", "val"], default= "val", type= str)
    parser.add_argument('--epochs', default= 1, type= int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    param =  setting_param()
    generate_LQ(source_dir= args.source_dir, deg_type= args.deg_type, param= param[args.deg_type], save_path= args.save_path, epochs= args.epochs, mode= args.mode)
