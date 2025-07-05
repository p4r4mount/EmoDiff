# [NeurIPS 2024] E-Motion: Future Motion Simulation via Event Sequence Diffusion

Arvix: [**E-Motion: Future Motion Simulation via Event Sequence Diffusion**](https://arxiv.org/abs/2410.08649).



## 🛠️ Requirements and Installation

* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:

```bash
git clone https://github.com/p4r4mount/E-Motion.git
cd E-Motion
conda env create -f environment.yml
```

## 📜 Datasets Preparation

Download the training data from [EventVOT](https://github.com/Event-AHU/EventVOT_Benchmark), and process the data with following command:

```bash
python utils/DataPreprocess.py --dataset_dir /path/to/dataset
```

## 🚅 Training

```bash
accelerate launch train.py \ 
	--num_processes num_processes \
	--main_process_port main_process_port \
	--config /path/to/config
```

## 💡 Pre-trained weights 

Google Drive: [Checkpoint](https://drive.google.com/drive/folders/1oeuoCFYm5sdmrCqHtulmiHpoEqNrg5ll?usp=drive_link)

## 🚀 Inference

Download some samples from [Google Drive](https://drive.google.com/drive/folders/1c8WvSVJ-kOG46-_9MbIAHCuv4nQWOTsO?usp=drive_link)，and run the following command for inference:

```bash
python predict.py --model_path /path/to/model/checkpoint \
                  --data_path /path/to/data/file.npy \
                  --output_path /path/to/output/directory
```



