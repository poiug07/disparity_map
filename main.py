import math
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from pathlib import Path

import postprocess
import psmnet
import sgbm
from psmnet import PSMNet

args = {}

def PSNRscore(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def main(method: str):
    if "psmnet" in  method:
        model = PSMNet(args["maxdisp"])
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        state_dict = torch.load(args["loadmodel"])
        model.load_state_dict(state_dict['state_dict'])

    # TODO: uncomment if running locally
    image_dirs = [str(p) for p in Path(args["imagedir"]).glob("*") if p.is_dir()]
    # OR: uncomment if running on kaggle
    # image_dirs = ['/kaggle/input/images-art', '/kaggle/input/images-dolls', '/kaggle/input/images-reindeer']

    total_psnr = 0
    for path in image_dirs:
        # TODO: uncomment if running locally
        leftpath, rightpath, truthpath, outputimg = f'{path}/view1.png', f'{path}/view5.png', f'{path}/disp1.png', f'{path}/pred.png'
        # OR: uncomment when if on kaggle
        # leftpath, rightpath, truthpath, outputimg = f'{path}/view1.png', f'{path}/view5.png', f'{path}/disp1.png', f'/kaggle/working/{path.split("-")[-1]}.png'
        truth = cv2.imread(truthpath, cv2.IMREAD_GRAYSCALE)
        sgbm_res = sgbm.run_stereo_sgbm(leftpath, rightpath)
        sgbm_inpaint_mean = postprocess.inpaint_mean(sgbm_res)
        
        if method=="sgbm-psmnet":
            disp = psmnet.run_psmnet(model, leftpath, rightpath, outputimg)
            psmnet_fill = postprocess.fill_in_missing(source=sgbm_inpaint_mean, dest=disp.copy())
            result = postprocess.mean_filter(psmnet_fill)
        elif method=="psmnet":
            disp = psmnet.run_psmnet(model, leftpath, rightpath, outputimg)
            result = postprocess.inpaint_mean(disp)
        else:
            result = sgbm_inpaint_mean

        # Replace output image with postprocessed output image
        cv2.imwrite(outputimg, result)

        psnr_score = PSNRscore(truth, result)
        total_psnr += psnr_score
        print(f"{path.split('/')[-1]}: {psnr_score}")
    print(f"Average PSNR score: {total_psnr/len(image_dirs)}")

if __name__=="__main__":
    # Parameter for PSMNet
    args["maxdisp"] = 192

    # TODO: Modify directories for images and model
    args["imagedir"] = "./images"
    args["loadmodel"] = "./models/finetune_248.tar"

    if len(sys.argv)<2:
        print("Argument missing: read README and see how to run.")
        sys.exit(1)
    method = sys.argv[1]
    if method not in ("sgbm","psmnet","sgbm-psmnet"):
        print("Unknown method. Please refer to README and see how to run.")
        sys.exit(1)

    if method!="sgbm":
        if not torch.cuda.is_available():
            raise ImportError("Need CUDA")
        torch.cuda.manual_seed(1)
    if cv2.ximgproc is None:
        raise ImportError("Dependency missing: opencv-contrib-python==4.7.0.72")

    torch.cuda.empty_cache() 
    main(method)