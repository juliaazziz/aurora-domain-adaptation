  <div align="center">
      <img src="assets/logo.png" align="center" alt="AURORA Logo" width="86" />
  </div>

  <div align="center">
  <h1 align="center">AURORA - Domain adaptation</h1>
  </div>

  <div align="center">
    <strong>Assessing a Domain-Adaptive Deployment Workflow for Selective Audio Recording in Wildlife Acoustic Monitoring</strong>
  </div>
  <div align="center">
    DCASE 2025.
  </div>

  <div align="center">
    <h3>
      <a href="https://gitlab.fing.edu.uy/aurora/aurora-ml">
        Original code
      </a>
      <span> | </span>
      <a href="https://gitlab.fing.edu.uy/aurora/aurora-fw">
        Firmware
      </a>
      <span> | </span>
      <a href="https://gitlab.fing.edu.uy/aurora/aurora-hw">
        Hardware
      </a>
    </h3>
  </div>


In this repository we publish the model checkpoints and code described in the following paper:

- **Title**: Assessing a Domain-Adaptive Deployment Workflow for Selective Audio Recording in Wildlife Acoustic Monitoring
- **Authors**: Julia Azziz, Josefina Lema, Lucía Ziegler, Leonardo Steinfeld, Martín Rocamora
- **Workshop**: Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events, (DCASE), Barcelona, Spain, 2025

The code found in this repository is a trimmed version of the [full project](https://gitlab.fing.edu.uy/aurora/aurora-ml), showing only what's relevant in the context of our DCASE2025 paper. For additional context you can head to the original repo.

## Abstract

Passive acoustic monitoring is a valuable tool for wildlife research, but scheduled recording often results in large volumes of audio, much of which may not be of interest. Selective audio recording, where audio is only saved when relevant activity is detected, offers an effective alternative. In this work, we leverage a low-cost embedded system that implements selective recording using an on-device classification model and evaluate its deployment for detecting penguin vocalization. 
To address the domain shift between training and deployment conditions (e.g. environment, recording device), we propose a lightweight domain adaptation strategy based on fine-tuning the model with a small amount of location-specific data. We replicate realistic deployment scenarios using data from two geographically distinct locations, Antarctica and Falkland Islands, and assess the impact of fine-tuning on classification and selective recording performance. Our results show that fine-tuning with location-specific data substantially improves generalization ability and reduces both false positives and false negatives in selective recording. These findings highlight the value of integrating model fine-tuning into field monitoring workflows, in order to improve the reliability of acoustic data collection.

## Getting started

The Python requirements are `python >= 3.10`. Create a virtual environment, either with conda or with pip. Activate the enviroment and install the required packages:

```sh
pip install -r requirements.txt
```

## Experiments

Fine-tuning settings are described in Section 4.3 of the paper.

To run a fine-tuning experiment use the following command inside the `experiments/` directory:

```sh
python run.py <data_dir> [<experiment_id>]
```

The script will run a K-fold cross-validation loop to determine the optimal layers to unfreeze, and then re-train the base model using the full dataset from the `--data_dir` directory.

To run this experiment several times (i.e. to ensure statistical robustness), you can use the following script:

```sh
./run.sh <data_dir>
```

## Citation

If you find this work useful, please cite our paper:

```bib
@inproceedings{Azziz2025,
    author = "Azziz, Julia and Lema, Josefina and Ziegler, Lucía and Steinfeld, Leonardo and Rocamora, Martín",
    title = "Assessing a Domain-Adaptive Deployment Workflow for Selective Audio Recording in Wildlife Acoustic Monitoring",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2025 Workshop (DCASE2025)",
    address = "Barcelona, Spain",
    month = "October",
    year = "2025",
}
```
