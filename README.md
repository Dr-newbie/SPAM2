# 🐽 SPAM : Spatial transcriptomics Predictor with self-supervised Alignment of Multimodalities

> **Context-aware spatial gene expression prediction from histology with multimodal self-supervised learning**
>
> Jaeyun Park, Dongsin Kim, Minsik Oh*  
> College of Data Technology, Myongji University

---

## 🧬 Overview 

**SPAM** is a multimodal framework for **predicting spatially resolved gene expression** from:

<p align='center'>
<img width="70%" alt="Image" src="https://github.com/user-attachments/assets/fba1dd38-7f7d-49b2-a380-be275b531fe2" />
</p>

- Histology image features (H&E patches, foundation models)
- Spatial coordinates (cell / spot locations)
- Gene expression profiles

The training pipeline consists of:

1. **Contrastive pretraining**  
   - Jointly learns representations of image, coordinates, and gene expression  
   - Uses a foundation image encoder + GCN + gene encoder

2. **Cross-attention + ZINB finetuning**  
   - Applies cross-attention between modalities (image ↔ coord, image ↔ gene)  
   - Merges attended features and reconstructs gene expression with a **ZINB decoder**

3. **Inference**  
   - Uses the finetuned model with biological context to predict gene expression for new sections  
   - Saves results (predicted expression, evaluation, plots, etc.)

---

## 📁 Repository Structure 

Please make "code" directory firstly if you clone this git repository

```text
SPAM/
├── models/          # Core model components
│   ├── Foundations.py      # Image foundation encoder wrapper (UNI, H-optimus, etc.)
│   ├── gene_encoder.py     # Gene expression encoder
│   ├── GCN_update.py       # Spatial GCN for coordinates
│   ├── contrastive.py      # Contrastive pretraining modules
│   └── ...
├── utils/           # Utilities
│   ├── dataset.py          # Dataset & dataloader
│   ├── graph_construction.py  # KNN graph building
│   ├── lora_utils.py       # (Optional) LoRA utilities
│   ├── loss_util.py        # ZINB loss, contrastive loss, etc.
│   └── ...
├── alignment.ipynb  # Example / debugging notebook for alignment
├── main.py          # Entry point for pretraining & finetuning
├── inference.py     # Entry point for inference (prediction)
└── README.md
```

--- 
## 💻 Environment & Installation 

```bash
git clone https://github.com/Dr-newbie/SPAM.git
cd SPAM

# (recommended) create conda env
conda env create -f enbironment.yml   

# Because we used Blackwell with cuda 12.9, you must install pytorch with below command (or install suitable CUDA & pytorch with your settings)
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
--- 
## 📊 Data preparation

Updates: we uploaded Human breast cancer dataset on our shared google drive link! 

Because of it's size (about 40GB) and data's characteristic, we were not able to upload toy data. 

Please visit the link and download the datasets! 👉 https://drive.google.com/drive/folders/1UXd_HEHfjjrtawK-ZQ6tEX3HQ6Va1NRX?usp=sharing

SPAM assumes three main inputs per section:
1. Spatial transcriptomics (.h5ad)
   *  e.g. Xenium / Visium h5ad containing cell/spot × gene matrix
   *  Path: --h5ad /path/to/section.h5ad
2. Patch-to-spot mapping CSV
   *  Each row = one image patch (tile)
   *  Typical columns:
      - x, y (pixel coordinates of patch center)
      - source_image (which WSI / tile image it came from)
      - cell_id or spot_id (to match with h5ad)
   *  Path: --csv /path/to/patch_map.csv
3. Image root directory
   * Directory containing cropped tiles or WSI images
   * Path: --root /path/to/image_root
     
Adapt these paths to your own preprocessing pipeline.

--- 
## 🚀 Usage
🎃 If you have your own public Xenium datasets , please use alignment.ipynb first🎃 

🧬 Or you can download the test sets here : https://www.10xgenomics.com/datasets/ffpe-human-breast-using-the-entire-sample-area-1-standard

1. ↔️ Contrstive learning
   * Run contrastive joint pretraining (image ↔ coord ↔ gene).
   
   ```bash
   python main.py pretrain \
     --h5ad /path/to/section.h5ad \
     --csv /path/to/patch_map.csv \
     --root /path/to/image_root \
     --enc_name uni_v1 \
     --mode joint \
     --epochs 50 \
     --batch_size 64 \
     --img_batch_size 256 \
     --save_dir ./checkpoints/pretrain \
     --device cuda \
     --amp
   ```
   Main arguments (typical):
      - h5ad : path to spatial transcriptomics AnnData file
      - csv : patch–spot mapping CSV
      - root : root directory of image tiles
      - enc_name : name of image foundation encoder (e.g. uni_v1, h-optimus-0, virchow ...
        - 👉 you must need your own hugging face token 👈
      - mode : pretraining mode (joint, img-gene, img-coord, etc. if supported)
      - epochs, batch_size, img_batch_size : training hyperparameters
      - save_dir : where to save pretraining checkpoints
      - device : cuda, cpu, or auto
      - amp : enable automatic mixed precision (optional flag)
   
   The best pretraining checkpoint will typically be saved under
   ./checkpoints/pretrain/ (e.g. best.pt).

3. 🎯 Cross-Attn + ZINB Finetuning
   * Finetune the model using cross-attention and a ZINB decoder for gene reconstruction.
   ```bash
   python main.py finetune \
     --h5ad /path/to/section.h5ad \
     --csv /path/to/patch_map.csv \
     --root /path/to/image_root \
     --enc_name uni_v1 \
     --pretrained_ckpt ./checkpoints/pretrain/best.pt \
     --epochs 30 \
     --batch_size 128 \
     --save_dir ./checkpoints/finetune \
     --device cuda
   ```
   Main arguments 
      - All shared arguments with pretraining: --h5ad, --csv, --root, --enc_name, …
      - --pretrained_ckpt : path to pretraining checkpoint (from step 5.1)
      - --epochs, --batch_size : finetuning hyperparameters
      - --save_dir : where to save the finetuned ZINB model
   
   Internally, the finetuning step:
      1. Builds image / coord / gene embeddings using pretrained encoders
      2. Applies cross-attention between modality pairs (e.g., image ↔ gene & image ↔ coords)
      3. Fuses attended embeddings with cross attention module
      4. Predicts gene expression with a ZINB decoder head
         
   You can download the pretrained model(Contrastive, just 10 epochs) here and put it in new folder checkpoints
   👉 https://drive.google.com/drive/folders/1UXd_HEHfjjrtawK-ZQ6tEX3HQ6Va1NRX?usp=sharing

5. 🔍 Inference (Gene Expression Prediction with biological context)
* After finetuning, run inference to predict gene expression on new data.
   ```bash
   python main.py inference \
     --h5ad /path/to/section.h5ad \
     --csv /path/to/patch_map.csv \
     --root /path/to/image_root \
     --enc_name uni_v1 \
     --ckpt ./checkpoints/finetune/best.pt \
     --out_dir ./results \
     --device cuda
   ```
   Main arguments (typical):
   
   - --h5ad : original ST h5ad (used for coordinates / metadata; can be empty for pure prediction)
   - --csv : patch–spot mapping CSV for the target section
   - --root : image root for the section
   - --enc_name : image encoder name (must match training)
   - --ckpt : path to the finetuned model checkpoint
   - --out_dir : directory to save:
      * Predicted expression matrix (.h5ad / .csv)
      * Evaluation metrics (if GT is available)
      * Optional plots (PCC distribution, scatter plots, etc.)
        
 you can use csv & use image root as we provide (spot_match.csv and patches.tar.gz respectively)   
 Please unzip the patches.tar.gz to make directory!!
 👉 https://drive.google.com/drive/folders/1UXd_HEHfjjrtawK-ZQ6tEX3HQ6Va1NRX?usp=sharing
 
 >  If you prefer to route inference through main.py instead
 >  (e.g. python main.py inference ...), you can simply add an
 >  inference subcommand in main.py and reuse the same arguments.
 
--- 
## 🪐 Tutorial
The jupyter notebook for usage & detailed experiments will update with:
   * Alignment pipelines
   * Gene expression prediction & measurements
   * Downstream task (Cell type clustering)
   * Gene expression map visualization
Stay tuned for our updates...🙏


--- 
## 📚 Citation
If you use SPAM in your research, please cite:
   > Jaeyun Park, Dongsin Kim, Minsik Oh*.
   > SPAM: Spatial transcriptomics Predictor with self-supervised Alignment of Multimodalities
   > College of Data Technology, Myongji University.
   > (Manuscript in preparation)
(Official BibTeX will be added once the paper is available.)


--- 
## ✉️ Contact
Feel free to contact us!!
For questions, issues, or collaboration:
   * Maintainer: Jaeyun Park
   * Email: banana9903@gmail.com
