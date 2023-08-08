# CS4186 HW2: Stereo Disparity Estimation

Check out report it is given in this repository and scores given in `results.txt`.

## How to run

Install required packages:

`pip install -r requirements.txt`

## Folder structure
```
root/
├── images/
│   ├── art/
│   │   ├── disp1.png
│   │   ├── view1.png
│   │   ├── view5.png
│   │   └── pred.png
│   └── ...
├── models/
│   └── finetune_248.tar
├── requirements.txt
├── postprocess.py
├── psmnet.py
├── sgbm.py
└── main.py
```
> Other directories or folders are not necessary to run program, but are to supplement report and submission.

Estimated disparity maps are saved in the corresponding directory with a name pred.png.  

## Available methods

Method `sgbm-psmnet` showed best results (you can refer to report).

* `sgbm` - run sgbm with inpainting and mean filtering.
* `psmnet` - run psmnet with inpainting and mean filtering.
* `sgbm-psmnet` - run method combining sgbm and psmnet results.

## Run disparity estimation

1. Modify some arguments in `main.py` if needed(not all systems support relative path):  
`args["imagedir"] = '/path/to/images'`  
`args["loadmodel"] = '/path/to/model'`  

2. Run on all images:  
`python main.py <method to run>`  

3. Output of `main.py` should look like:
```
images-art: 21.193373713365897
images-dolls: 22.700798084164965
images-reindeer: 21.072991080149762
Average PSNR score: 21.655720959226873
```