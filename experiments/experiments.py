import os, sys
sys.path.append("..")
sys.path.append("../training")

import gc
import tqdm
import pickle
import numpy as np
import tensorflow as tf
    
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model

from trainer import Trainer
from utils.printer import *
from utils.tfrecord_handler import get_dataset_length

N_UNFREEZE_LAYERS = [None, 1, 2] # 67 total layers in our resnet
TRAIN_SIZE = 0.3
VAL_SIZE = 0.7
N_SPLITS = 10
N_FOLDS = 5

class ExperimentRunner():
    def __init__(self, dataset=None, n_splits=N_SPLITS, n_folds=N_FOLDS, modelfile="../training/out/model.keras"):
        """
        Initialize the ExperimentRunner instance.
        """
        if dataset is not None:
            self.dataset = dataset
            self.ds_len = get_dataset_length(dataset)
        else:
            raise ValueError("No dataset provided")
            
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.modelfile = modelfile

        self.train_size = TRAIN_SIZE
        self.val_size = VAL_SIZE
        self.train_len = int(self.train_size * self.ds_len)

        self.outfolder = "out/"
        self.metrics = {}
        self.history = {}

    def run(self, train_ds, val_ds, train_splits, split_id, unfreeze_layers=N_UNFREEZE_LAYERS, seed=42):
        """
        Run the experiment with multiple 30-70 splits and training fractions.
        """
        self.exp_id = split_id
        self.metrics_filename = f"{self.outfolder}/metrics_{self.exp_id}.pkl"
        self.history_filename = f"{self.outfolder}/history_{self.exp_id}.pkl"        

        # Loop over 70-30 splits
        for frac in tqdm.tqdm(train_splits, desc="Training fractions"):
            sub_train_len = int(self.train_len * frac)

            sub_train_ds = train_ds.take(sub_train_len)
            indexed_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(tf.range(sub_train_len)), sub_train_ds))
            idx = range(sub_train_len)

            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed + split_id)

            layer_scores = {}
            for n_layers in tqdm.tqdm(unfreeze_layers, desc="Fine-tuning"):
                scores = []
                
                # Perform K-Fold CV, this will find the best number of layers to unfreeze
                for fold_id, (train_idx, val_idx) in enumerate(kf.split(idx)):
                    # get sub_train_ds elements based on index
                    train_idx_set = tf.constant(train_idx, dtype=tf.int32)
                    val_idx_set = tf.constant(val_idx, dtype=tf.int32)

                    # Filter dataset based on indices
                    fold_train = indexed_ds.filter(lambda i, data: tf.reduce_any(tf.equal(i, train_idx_set)))
                    fold_train = fold_train.map(lambda i, data: data)
                    fold_val   = indexed_ds.filter(lambda i, data: tf.reduce_any(tf.equal(i, val_idx_set)))
                    fold_val = fold_val.map(lambda i, data: data)

                    trainer = Trainer(model_name="resnet18", epochs=20)
                    model = load_model(self.modelfile)
                    trainer.model = model
                    trainer.ft_train_ds = fold_train
                    trainer.ft_val_ds = fold_val

                    trainer.trained_modelfile = f"{self.outfolder}/tmp.keras"
                    trainer.fine_tune(None, N=n_layers, set_records=False)
                    metrics = trainer.evaluate(verbose=0)
                    scores.append(metrics["val"].get('accuracy', 0))

                    # Cleanup
                    os.remove(trainer.trained_modelfile)     
                    gc.collect()

                layer_scores[n_layers] = np.mean(scores)
                
            # Select best number of layers for this sub_train_data based on CV results
            best_layers = max(layer_scores, key=layer_scores.get)

            # Fine-tune on full training set using best layers
            trainer = Trainer(model_name="resnet18", epochs=20)
            trainer.model = model
            trainer.ft_train_ds = sub_train_ds
            trainer.ft_val_ds = val_ds  # always evaluate on full validation set
            trainer.trained_modelfile = f"{self.outfolder}/model_exp{self.exp_id}_split{split_id}_frac{frac:.2f}_layers{best_layers}.keras"
            
            hist = trainer.fine_tune(None, N=best_layers, set_records=False)
            mets = trainer.evaluate(verbose=0)

            key = (split_id, frac, best_layers)
            self.history[key] = hist
            self.metrics[key] = mets

        self.save_results()

        return self.history, self.metrics

    def save_results(self):
        """
        Save the results to a file.
        """
        with open(self.history_filename, 'wb') as f:
            pickle.dump(self.history, f)

        with open(self.metrics_filename, 'wb') as f:
            pickle.dump(self.metrics, f)