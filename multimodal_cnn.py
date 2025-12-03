import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0, VGG16, Xception, MobileNetV2

from tensorflow.keras import optimizers, callbacks

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import itertools

import keras
from keras.applications import ResNet50, Xception
from keras.models import Sequential, Model
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AveragePooling2D,GlobalMaxPooling2D
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras import regularizers
from keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score, matthews_corrcoef

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import pickle
from datetime import datetime

np.random.seed(123)
import time


from stratified_cross_validator import StratifiedCrossValidator
from utils.data_preparation import handle_missing_values

class MultimodalCNNFramework:
    """
    Framework CNN multimodal combinant images et métadonnées 
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=7, 
                 batch_size=32, epochs=100, learning_rate=0.001,
                 metadata_cols=None):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.metadata_cols = metadata_cols or []
        
        self.architectures = {}
        self.training_history = {}
        self.evaluation_results = []
        
        # initialisation des gpt si disponible
        self._setup_gpu()
        
    def _setup_gpu(self):
        """
        Configure l'utilisation du GPU si disponible.
        """
        devices_gpu = tf.config.experimental.list_physical_devices('GPU')
        
        if devices_gpu:
            try:
                for device in devices_gpu:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f" GPU configuré: {len(devices_gpu)} GPU(s) disponible(s)")
            except RuntimeError as e:
                print(f"  Erreur configuration GPU: {e}")
   
   
    def prepare_metadata(self, X_train, X_test, X_val, target_col: str, transform_columns: list, save_dir: str = "encoders", fold_number: int = 0):

        os.makedirs(f'{save_dir}/fold_{fold_number}', exist_ok=True)

        # Encodage de la cible
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(X_train[target_col])
        y_val = label_encoder.transform(X_val[target_col])
        
        
        y_test = label_encoder.transform(X_test[target_col])

        # Gestion des valeurs manquantes
        X_train = handle_missing_values(X_train)
        X_val = handle_missing_values(X_val)
        X_test = handle_missing_values(X_test)

        # One-hot encoding
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        col_train = onehot_encoder.fit_transform(X_train[transform_columns])
        col_val = onehot_encoder.transform(X_val[transform_columns])
        col_test = onehot_encoder.transform(X_test[transform_columns])

        # Standardisation de l'âge
        scaler_age = StandardScaler()
        age_train = scaler_age.fit_transform(X_train[['age']])
        age_val = scaler_age.transform(X_val[['age']])
        age_test = scaler_age.transform(X_test[['age']])

        # Combinaison des features
        X_train_tab = np.concatenate([col_train, age_train], axis=1)
        X_val_tab = np.concatenate([col_val, age_val], axis=1)
        X_test_tab = np.concatenate([col_test, age_test], axis=1)

        # Sauvegarde
        joblib.dump(label_encoder, os.path.join(f'{save_dir}/fold_{fold_number}/', f"{fold_number}_label_encoder.pkl"))
        joblib.dump(onehot_encoder, os.path.join(f'{save_dir}/fold_{fold_number}/', f"{fold_number}_onehot_encoder.pkl"))
        joblib.dump(scaler_age, os.path.join(f'{save_dir}/fold_{fold_number}/', f"{fold_number}_scaler_age.pkl"))
    
        X_train.to_csv(os.path.join(f'{save_dir}/fold_{fold_number}/', f"X_train.csv"), index=False)
        X_val.to_csv(os.path.join(f'{save_dir}/fold_{fold_number}/', f"X_val.csv"), index=False)
        X_test.to_csv(os.path.join(f'{save_dir}/fold_{fold_number}/', f"X_test.csv"), index=False),

        # Retour
        X_dict = {"train": X_train_tab, "val": X_val_tab, "test": X_test_tab}
        y_dict = {"train": y_train, "val": y_val, "test": y_test}
        transformers = {
            "label_encoder": label_encoder,
            "onehot_encoder": onehot_encoder,
        "scaler_age": scaler_age
        }

        return X_dict, y_dict, transformers
    
    
    def preprocess_image_data_with_pretrained(self, train_df, test_df, val_df, preprocess_func):
        """
            Prétraitement des images déjà chargées (présentes dans la colonne 'image').

        Args:
            train_df, test_df, val_df : DataFrames contenant une colonne 'image'
            preprocess_func : fonction Keras (ex : preprocess_input)

        Returns:
            x_train, x_test, x_val : tableaux numpy prétraités
        """
    # Stack les images (shape attendue : (N, H, W, 3))
        x_train = np.stack(train_df["image"].values)
        x_test = np.stack(test_df["image"].values)
        x_val = np.stack(val_df["image"].values)

        # Appliquer le prétraitement
        if preprocess_func:
            x_train = preprocess_func(x_train.astype(np.float32))
            x_test = preprocess_func(x_test.astype(np.float32))
            x_val = preprocess_func(x_val.astype(np.float32))

        return x_train, x_test, x_val


         
        
    def create_multimodal_resnet(self, metadata_dim, name="MultimodalResNet"):
        """Crée un modèle ResNet multimodal."""
        # Branche image
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(512, activation='relu', name='image_features')(x)
        
        # Branche métadonnées
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)
            
            # Fusion des branches
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)
            
            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
            # Modèle image uniquement
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)
        
        self.architectures[name] = model
        print(f"model ok {name} créé - Paramètres: {model.count_params():,}")
        return model
    

    def create_multimodal_xception(self, metadata_dim, name="MultimodalXception"):
        """Crée un modèle Xception multimodal."""

        # Branche image (Xception)
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  
            layer.trainable = False

        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(512, activation='relu', name='image_features')(x)

        # Branche métadonnées
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)

        # Fusion des branches
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)

            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
            # Modèle image uniquement
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)

        self.architectures[name] = model
        print(f"model ok {name} créé - Paramètres: {model.count_params():,}")
        return model
    

    def create_multimodal_mobilenet(self, metadata_dim, name="MultimodalMobileNet"):
            

   
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # Gèle les premières couches
            layer.trainable = False

        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(512, activation='relu', name='image_features')(x)

    
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)

        
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)

            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
        
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)

        self.architectures[name] = model
        print(f"model ok {name} créé - Paramètres: {model.count_params():,}")
        return model


    
    def compile_models(self):
        """Compile tous les modèles."""
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        for name, model in self.architectures.items():
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        
        print("----------------  Tous les modèles compilés--------------------")
        

    def create_callbacks(self, model_name, fold, save_dir="models", monitor='val_loss'):
        os.makedirs(f'{save_dir}/fold_{fold}/', exist_ok=True)
    
        callbacks_list = [
            callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
            ),
        
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
        
            callbacks.ModelCheckpoint(
                filepath=f"{save_dir}/fold_{fold}/{model_name}_fold_{fold}_best.keras",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
        
            callbacks.CSVLogger(
                filename=f"{save_dir}/fold_{fold}/{model_name}_fold_{fold}_training.csv"
            )
        ]
    
        return callbacks_list

    
    def train_model(self, model_name, train_data, val_data, test_data, y_true, fold):
        """Entraîne un modèle multimodal."""
        if model_name not in self.architectures:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        model = self.architectures[model_name]
        callbacks_list = self.create_callbacks(model_name, fold)
        
        print(f"\n Entraînement {model_name} - Fold {fold}")
        #print("Val generator ",val_data)
        print(f"   Train batches: {len(train_data)}")
        print(f"   Val batches: {len(val_data)}")
        
        start_time = time.time()
        
        # Entraînement
        history = model.fit(
            train_data,
            epochs=self.epochs,
            validation_data=val_data,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Stockage
        history_key = f"{model_name}_fold_{fold}"
        self.training_history[history_key] = {
            'history': history.history,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        self.evaluate_on_test(model_name,model, test_data,y_true ,fold=fold, save_path=f'models/{model_name}fold_{fold}/')
        self.summarize_results(save_path=f'models/{model_name}/fold_{fold}/')
        
        print(f" Entraînement terminé en {training_time:.2f}s")
        model.save(f'Model_{model_name}_fold_{fold}.keras')
        #loaded_model = tf.keras.models.load_model(f'Model_{model_name}_fold_{fold}.keras')
        #print(loaded_model)

        return history
    
    
    
    def evaluate_on_test(self, model_name, model, test_data, y_true, fold, save_path="results"):
        """
            Évalue le modèle sur les données de test et sauvegarde les métriques MCC et F1.
        """
        os.makedirs(save_path, exist_ok=True)

        # Prédictions
        y_pred_proba = model.predict(test_data)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calcul des métriques
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Stockage des résultats
        if not hasattr(self, "evaluation_results"):
            self.evaluation_results = []

        self.evaluation_results.append({
            "model": model_name,
            "fold": fold,
            "mcc": mcc,
            "f1_score": f1
        })

        print(f"Fold {fold} — F1: {f1:.4f}, MCC: {mcc:.4f}")

        # Sauvegarde à chaque fold
        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(os.path.join(save_path, f"{model_name}_evaluation_folds.csv"), index=False)



    def summarize_results(self, save_path="results"):
        """
        Calcule et sauvegarde les statistiques finales des modèles.
        """
        os.makedirs(save_path, exist_ok=True)
        if not hasattr(self, "evaluation_results"):
            print("Aucun résultat à résumer.")
            return
    
        df = pd.DataFrame(self.evaluation_results)
        summary = df.groupby("model")[["f1_score", "mcc"]].agg(["mean", "std"])
        summary.to_csv(os.path.join(save_path, "summary_metrics.csv"))
        print("\nRésumé final :\n", summary)
        
        
        
        
    def create_multimodal_densenet(self, metadata_dim, name="MultimodalDenseNet"):
        """Crée un modèle DenseNet multimodal."""
        # Branche image
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(512, activation='relu')(x)
        
        # Branche métadonnées
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)
            
            # Fusion
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)
            
            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)
        
        self.architectures[name] = model
        print(f" {name} créé - Paramètres: {model.count_params():,}")
        return model
    
    def create_multimodal_efficientnet(self, metadata_dim, name="MultimodalEfficientNet"):
        """Crée un modèle EfficientNet multimodal."""
        # Branche image
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.6)(x)
        image_features = layers.Dense(512, activation='relu')(x)
        
        # Branche métadonnées
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)
            
            # Fusion
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)
            
            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)
        
        self.architectures[name] = model
        print(f" {name} créé - Paramètres: {model.count_params():,}")
        return model
    
    def create_multimodal_vgg(self, metadata_dim, name="MultimodalVGG"):
        """Crée un modèle VGG multimodal."""
        # Branche image
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(512, activation='relu')(x)
        
        # Branche métadonnées
        if metadata_dim > 0:
            metadata_input = layers.Input(shape=(metadata_dim,), name='metadata_input')
            meta_x = layers.Dense(128, activation='relu')(metadata_input)
            meta_x = layers.BatchNormalization()(meta_x)
            meta_x = layers.Dropout(0.3)(meta_x)
            meta_x = layers.Dense(64, activation='relu')(meta_x)
            meta_features = layers.BatchNormalization()(meta_x)
            
            # Fusion
            combined = layers.concatenate([image_features, meta_features])
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.4)(combined)
            
            outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            model = models.Model(inputs=[image_input, metadata_input], outputs=outputs)
        else:
            x = layers.Dense(256, activation='relu')(image_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            model = models.Model(inputs=image_input, outputs=outputs)
        
        self.architectures[name] = model
        print(f" {name} créé - Paramètres: {model.count_params():,}")
        return model
