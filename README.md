# Deepfake-Health





## FaceFusion
### Original FaceFusion github link
https://github.com/facefusion/facefusion

### Implementation command
```
python facefusion.py --video 'Source video directory' --savepath 'saving directory' --source 'targetted face image directory' --onnx
```

### License
https://docs.facefusion.io/introduction/licenses


## Diff2Lip
### Original Diff2lip github link
https://github.com/soumik-kanad/diff2lip

### Diff2Lip Checkpoint
Please download the checkpoint from the Diff2Lip_ckpt folder using the link below.

https://osf.io/q4p3v/

### Implementation command
```
./scripts/inference_single_video.sh
```

### License
Except where otherwise specified, the text/code on Diff2Lip repository by Soumik Mukhopadhyay (soumik-kanad) is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). It can be shared and adapted provided that they credit us and don't use our work for commercial purposes.

### Citation
bibtex for Diff2lip by Soumik Mukhopadhyay

@InProceedings{Mukhopadhyay_2024_WACV,
    author    = {Mukhopadhyay, Soumik and Suri, Saksham and Gadde, Ravi Teja and Shrivastava, Abhinav},
    title     = {Diff2Lip: Audio Conditioned Diffusion Models for Lip-Synchronization},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5292-5302}
}
