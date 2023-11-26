
  
# PIER  
  
Implementation for Potentially Idiomatic Expression Representation (PIER) model  
  
  
## Description  
  
This repo contains the implementation for the Potentially Idiomatic Expression Representation (PIER) model from the paper: Unified Representation for Non-compositional and Compositional Expressions.  
  
The paper is published in EMNLP2023 Findings. Please refer to the paper for more details.  
  
## Getting Started    
  
### Dependencies  
  
We recommend running the model training and evaluation in a Python virtual environment.  
The dependencies and required packages are listed in `requirements.txt`.  
  
Note that though our work utilized [AdapterHub](https://github.com/adapter-hub)'s adapter implementation.   We maintain a local copy of the AdapterHub source code in which we additionally implemented the `Fusion` module.  Hence, AdapterHub is not part of the requirements.  
  
### Data  
To run the model training and evaluation, supporting data (e.g., feature maps, training data, etc.) must be downloaded and place to the proper locations.  
  
Please download data and put them into the following two directories:  
* `./data`: Download and extract data from [here](https://drive.google.com/file/d/1tUwo5aL4o3ioJzppwcpMphYAzOnBnwfm/view?usp=sharing).  
* `./fusion_analysis`: Download and extract data from [here](https://drive.google.com/file/d/1PCkCHi6lBIid6rW5bLnRijZyp2yixPac/view?usp=sharing).  
  
### Model Checkpoints  
PIER model utilizes pretrained [GIEA](https://arxiv.org/abs/2207.03679) model checkpoint as part of its architecture. So, before training the PIER model, you need to download the GIEA's checkpoints.  
* GIEA model checkpoints: download [link](https://drive.google.com/file/d/1ZkKzQCnFCsdAN438l4worfK4kuSAJeoK/view?usp=sharing).
* Extract the checkpoint and place it under the direction: `./checkpoints/bart-adapters_non-compositional_magpie_random-GIEA`  
  
We also provide our the checkpoints of best-performing model from the paper, PIER+. Download the trained checkpoint and you can perform IE embedding and PIER+ intrinsic evaluation directly without having to train the model yourself.  
* PIER model checkpoints: download [link](https://drive.google.com/file/d/14NzSUxRy2x3Vz0fnzLUpSSGNrPzrK57d/view?usp=sharing).
* Extract the checkpoint and place it under the direction: `./checkpoints/bart-adapters_fusion_magpie_random-PIER`  
  
## Key Components  

### PIER model  
The PIER+ model is implemented in the `BartAdapterCombined` class at `src/model/bart_adapters.py`

The PIER model used for NLU classification tasks (i.e., paraphrase identification and sentiment classification) is implemented in the `BartAdapterCombinedCLS` class at `src/model/bart_adapters.py` 

### Downstream classifiers
The classifier for the extrainsic evaluations (i.e., idiom processing tasks and NLU classification tasks) are implemented in the following locations: 
* IE sense classification (SenseCLF): `./src/classifiers/literal_idiom_classifier.py`
* PIE span detection (SpanDET): `./src_idiom_detect/model/bilstm.py`
* Sentiment classification (SentCLF): `./src/classifiers/sentiment_classifier.py`
* Paraphrase identification (ParaID): `./src/classifiers/paraphrase_identifier.py`
  
## Model Training  
To follow the paper, we first train the PIER+ model and then we train a classifier for each of the downstream applications. 

### Training PIER model 
To train the PIER model: 
1. Set the configuration in `config.py`
   * The setting here includes the name for the checkpoints, data path (already set), and common training configs, e.g., batch size and number of epochs. Please refer to the paper for the exact settings that the PIER+ used.
2. Run `train_pier_model.py`

Note that a trained PIER+ checkpoint is provided [here](https://drive.google.com/file/d/14NzSUxRy2x3Vz0fnzLUpSSGNrPzrK57d/view?usp=sharing). Download the checkpoint here directly to avoid training. 

### Training downstream classifiers
Similar to training the PIER model, first we need to set configuration appropriately in `config.py`. Please refer to the comments in the file for more details. 

The training entrypoints for the downstream classifiers are the following: 
* IE sense classification (SenseCLF): `./train_cls_model.py`
* PIE span detection (SpanDET): `./train_det_model.py`
* Sentiment classification (SentCLF): `./train_sc_model.py`
* Paraphrase identification (ParaID): `./train_pi_model.py`


## Model Evaluation  

### Intrinsic evaluations

With a trained PIER+ checkpoint (newly trained or downloaded), we can perform the intrinsic evaluation as described in the paper. 

1. Generate the PIE embeddings
   * Run `EVALUATION_PIER_Embedding_Generation.ipynb` to generate the PIE embeddings with the trained PIER+ model. 
   * The script also allows the exploration of the similar IEs in the PIER+'s embedding space.  
2. Compute the H-Score and CosDist score.
   * Run `EVALUATION_Intrinsic_H-score_and_CosDist.ipynb`
   * This script will read the IE meaning groups and then run clustering to compute the H-score and CosDist score. 
3. Compute the DiffSim score. 
   * Run `EVALUATION_Intrinsic_DiffSim.ipynb`
   * This script will generate both literal and idiomatic PIE embeddings wih the PIER+ model and then compute the DiffSIm scores. 

   
Please refer to the paper for more details on the Intrinsic evaluations. 


### Extrinsic evaluations

Once the downstream classifiers are trained, run the following scripts to gather the extrinsic evaluation results.
* IE sense classification (SenseCLF): `./EVALUATION_IdiomaticLiteralClassification.ipynb`
* PIE span detection (SpanDET): `./EVALUATION_SpanDetection.ipynb`
* Sentiment classification (SentCLF): `./EVALUATION_SentimentClassification.ipynb`
* Paraphrase identification (ParaID): `./EVALUATION_ParaphraseIdentification.ipynb`

  
## Authors  
   
Ziheng Zeng (zzeng13@illinois.edu)  
  
## Version History  
  
* 0.2  
    * Added evaluation example for intrinsic evaluations.  
    * Added online storage location for data and checkpoints.  
* 0.1  
  * Initial Release  
  
## License  
  
This project is licensed under the MIT License.