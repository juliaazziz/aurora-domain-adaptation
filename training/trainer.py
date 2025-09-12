"""
@file trainer.py
@brief Class for training, fine-tuning, and quantizing models.
"""

import sys
sys.path.append("../")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from training.resnet18 import *
from utils.printer import *
from utils.tfrecord_handler import TFRecordHandler
from utils.frontend_settings import frontend_settings

class Trainer:
    def __init__(self, model_name="resnet18", dataset="../data/falklands", batch_size=64, epochs=35, dropout=0.0):
        """
        Initializes the trainer.

        Arguments:
            model_name (str): Model name
            dataset (str): Name of dataset
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            dropout (float): Dropout rate
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout

        self.records_path = f"{dataset}/records/"

        self.train_record = self.records_path + "train.tfrecord"
        self.test_record  = self.records_path + "test.tfrecord"
        self.val_record   = self.records_path + "val.tfrecord"

        self.handler = TFRecordHandler(self.records_path,
                                       f"{dataset}/metadata.csv",
                                       frontend_settings)
        self.train_ds = self.handler.create_batched_dataset(self.train_record, batch_size=self.batch_size)
        self.test_ds = self.handler.create_batched_dataset(self.test_record, batch_size=self.batch_size)
        self.val_ds = self.handler.create_batched_dataset(self.val_record, batch_size=self.batch_size)

        for spec, _ in self.train_ds.take(1):
            self.input_shape = spec.shape + (1,)

        self.initial_modelfile = f"out/{self.model_name}/initial_{self.model_name}.keras"
        self.trained_modelfile = f"out/{self.model_name}/trained_{self.model_name}.keras"
        self.tflite_modelfile = f"out/{self.model_name}/tflite_{self.model_name}.tflite"

    def _build_model(self):
        """
        Builds the model based on the selected architecture.
        """
        print_msg(LogLevel.INFO, f"Building {self.model_name} model...")

        if self.model_name == "resnet18":
            base_model = ResNet18(self.input_shape)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")
        
        model = keras.models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(32),
            layers.Dropout(self.dropout),
            layers.Activation("relu"),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

        return model

    def train(self):
        """
        Trains the model from scratch.
        """
        print_msg(LogLevel.INFO, f"Training {self.model_name} model...")

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=4, min_lr=1e-6)

        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            callbacks=[early_stopping, reduce_lr],
            validation_data=self.val_ds
        )

        self.model.save(self.initial_modelfile)

        return history

    def fine_tune(self, records_path, batch_size=16, N=None, set_records=True, verbose=True):
        """
        Loads and fine-tunes a pre-trained model by unfreezing the top layers.
        """
        if set_records:
            self.ft_train_record = records_path + "/records/train.tfrecord"
            self.ft_val_record   = records_path + "/records/val.tfrecord"

            self.ft_handler = TFRecordHandler(records_path,
                                            records_path + "/metadata.csv",
                                            frontend_settings)
            self.ft_train_ds = self.handler.create_batched_dataset(self.ft_train_record, batch_size=batch_size)
            self.ft_test_ds = self.handler.create_batched_dataset(self.ft_test_record, batch_size=batch_size)
            self.ft_val_ds = self.handler.create_batched_dataset(self.ft_val_record, batch_size=batch_size)

        # freeze trained layers
        for layer in self.model.layers[0].layers:
            layer.trainable = False

        if N is not None:
            for layer in self.model.layers[0].layers[-N:]:
                layer.trainable = True

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(1e-4),
            metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=4, min_lr=1e-6)

        history = self.model.fit(
            self.ft_train_ds.repeat(batch_size),
            epochs=self.epochs,
            callbacks=[early_stopping, reduce_lr],
            validation_data=self.ft_val_ds.repeat(batch_size),
            verbose=verbose
        )

        self.model.save(self.trained_modelfile)

        return history

    def evaluate(self, after_finetuning=True, verbose=True):
        """
        Evaluates the model on the training and validation sets.
        Returns a dictionary containing loss, accuracy, AUC, PRAUC, and F1-score.
        """

        if after_finetuning:
            train_dataset = self.ft_train_ds
            val_dataset = self.ft_val_ds
        else:
            train_dataset = self.train_ds
            val_dataset = self.val_ds

        train_metrics = self._compute_metrics(train_dataset, verbose)
        val_metrics = self._compute_metrics(val_dataset, verbose)

        return {"train": train_metrics, "val": val_metrics}

    def _compute_metrics(self, dataset, verbose=True):
        y_true, y_pred, y_scores = [], [], []
        
        for x, y in dataset:
            preds = self.model.predict(x, verbose=verbose)
            y_scores.extend(preds)
            y_true.extend(y.numpy())
        
        y_pred = [score > 0.5 for score in y_scores]

        auc = roc_auc_score(y_true, y_scores)
        prauc = average_precision_score(y_true, y_scores)
        f1 = f1_score(y_true, y_pred)

        loss, accuracy = self.model.evaluate(dataset, verbose=verbose)

        return {"loss": loss, "accuracy": accuracy, "auc": auc, "prauc": prauc, "f1_score": f1}