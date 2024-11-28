# MMDuet
[![Static Badge](https://img.shields.io/badge/🤗Model-MMDuet-yellow)](https://huggingface.co/wangyueqian/MMDuet)
[![Static Badge](https://img.shields.io/badge/🤗Dataset-MMDuetIT-yellow)](https://huggingface.co/datasets/wangyueqian/MMDuetIT)
[![arXiv](https://img.shields.io/badge/arXiv-2411.17991-b31b1b.svg)](https://arxiv.org/abs/2411.17991)


Official implementation of paper *VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format*

# Introduction

[![Watch video on Youtube](http://img.youtube.com/vi/n1OybwhQvtk/0.jpg)](https://www.youtube.com/watch?v=n1OybwhQvtk)

Video also available on [Bilibili (゜-゜)つロ干杯~](https://www.bilibili.com/video/BV1nwzGYBEPE)

MMDuet is a VideoLLM implemented in the *video-text duet interaction format*, which treats the video stream as a role in the conversation akin to the user and the assistant. Under this interaction format, the video is continuously played and input to the model frame-by-frame. Both the user and model can insert their text messages right after any frame during the video play. When a text message ends, the video continues to play, akin to the show of two performers in a duet. 

**This not only ensures a timely response for video comprehension, but also improves the performance on many time-sensitive video-text multimodal tasks, such as temporal video grounding, highlight detection, and dense video captioning.**

# Installation
1. Create conda environment and use pip to install some packages
```shell
pip clone https://github.com/yellow-binary-tree/MMDuet
cd MMDuet

conda create -n mmduet python=3.10
conda activate mmduet
pip install --upgrade pip
pip install -r requirements.txt
```

2. Install llava following the instructions in [https://github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
```

3. Install flash-attention following the instructions in [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). If you have difficulties installing it, add `--attn_implementation sdpa` in every command to use the sdpa implementation of transformer attention for train or inference.

4. Download MMDuet checkpoints from HuggingFace: [https://huggingface.co/wangyueqian/MMDuet](https://huggingface.co/wangyueqian/MMDuet) and put the files under folder `./outputs/mmduet`.

# Demo
To launch a Gradio demo: `python -m demo.app --lora_pretrained outputs/mmduet`

# Inference
## Download model and data

- Download our data annotation for training (MMDuetIT) and evaluation from [wangyueqian/MMDuetIT](https://huggingface.co/datasets/wangyueqian/MMDuetIT) and put them in `datasets` folder.

- Download the videos, and link each video folder to `datasets/${DATASET_NAME}/videos`. Here we list recommended video download links, while you can also download from other sources:
  - YouCook2: [https://opendatalab.com/OpenDataLab/YouCook2](https://opendatalab.com/OpenDataLab/YouCook2)
  - Shot2Story: [https://huggingface.co/mhan/shot2story-videos](https://huggingface.co/mhan/shot2story-videos)
  - Charades: [https://prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades)
  - QVHighlights: [https://github.com/jayleicn/moment_detr/blob/main/data/README.md](https://github.com/jayleicn/moment_detr/blob/main/data/README.md)

- Download [paraphrase-en.gz](https://github.com/lichengunc/refer/raw/refs/heads/master/evaluation/meteor/data/paraphrase-en.gz) (59MB) which is used for dense video captioning evaluation. Put this file at `test/dvc/metrics/data/paraphrase-en.gz`

## Inference and evaluation
Scripts to inference on all benchmarks are listed in `./scripts/inference/`.

**WARNING**: Each script file contains many steps for inference and evaluation. DO NOT directly run these script files. Instead, read the contents of these files carefully and run them step by step.

- YouCook2 dense video captioning: `./scripts/inference/youcook2.sh`
- Shot2Story-MAGQA-39k multi-answer grounded video question answering (MAGQA): `./scripts/inference/magqa.sh`
  - **Note**: To save compute, we do not calculate the similarity score between the pred answer and the gold answer if the pred time is not in the gold timespan. We simply set this score to 1 in the score matrix of evaluator_output. These scores are not used in calculating and do not affect the final metric (in-span score).
- Charades-STA temporal video grounding: `./scripts/inference/charades.sh`
- QVHighlights highlight detection: `./scripts/inference/qvh.sh`


# Training

- If you want to reproduce the training process, you also need to download the training data. Download the videos, and link each video folder to `datasets/${DATASET_NAME}/videos`. Here we list recommended video download links, while you can also download from other sources:
  - COIN: [https://huggingface.co/datasets/WHB139426/coin](https://huggingface.co/datasets/WHB139426/coin)
  - HiREST: [https://github.com/j-min/HiREST](https://github.com/j-min/HiREST)
  - DiDeMo: [https://github.com/LisaAnne/TemporalLanguageRelease](https://github.com/LisaAnne/TemporalLanguageRelease)
  - QueryD: [https://www.robots.ox.ac.uk/~vgg/data/queryd/](https://www.robots.ox.ac.uk/~vgg/data/queryd/)

Run `./scripts/train.sh`.

When running training code for the first time, the dataset code will traverse all videos of the training dataset and stat the frame rate, duration and number of frames of the videos, and store this information in `datasets/${dataset_name}/videos_metadata.json`. This can take quite a long time.
Considering that videos downloaded from different sources may be slightly different, in order to ensure that the videos are correctly loaded, we do not include this metadata information in our data release.

# Acknowledgment
The following projects has been of great help to this work:
- [VideoLLM-online](https://github.com/showlab/VideoLLM-online) for providing codebase we built upon,
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) for providing awesome multi-modal foundation models,
- [Shot2Story](https://github.com/bytedance/Shot2Story) for providing high-quality clip-level video captions.

# Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{wang2024mmduet,
      title={VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format}, 
      author={Yueqian Wang and Xiaojun Meng and Yuxuan Wang and Jianxin Liang and Jiansheng Wei and Huishuai Zhang and Dongyan Zhao},
      year={2024},
      eprint={2411.17991},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17991}, 
}
```
