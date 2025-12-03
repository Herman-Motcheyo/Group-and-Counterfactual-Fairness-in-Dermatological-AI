
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xceptopn_preprocess


from utils.utils import make_multimodal_dataset, save_history_to_csv
import numpy as np 
import pandas as pd 

from PIL import Image

def get_model_config():
    """
    Retourne la configuration pour chaque modèle avec ses paramètres spécifiques.
    """
    model_configs = {
        'xception': {
            'create_function': 'create_multimodal_xception',
            'model_name': 'MultimodalXception',
            'preprocess_func': xceptopn_preprocess,
            'input_size': (224, 224)  
        },
        'mobilenetv2': {
            'create_function': 'create_multimodal_mobilenet',
            'model_name': 'MultimodalMobileNet',
            'preprocess_func': mobilenet_preprocess,
            'input_size': (224, 224)  
        },
        'resnet': {
            'create_function': 'create_multimodal_resnet',
            'model_name': 'MultimodalResNet',
            'preprocess_func': resnet_preprocess,
            'input_size': (224, 224)  
        },
        'vgg': {
            'create_function': 'create_multimodal_vgg',
            'model_name': 'MultimodalVGG',
            'preprocess_func': vgg_preprocess,
            'input_size': (224, 224)  
        },
        'efficientnet': {
            'create_function': 'create_multimodal_efficientnet',
            'model_name': 'MultimodalEfficientNet',
            'preprocess_func': efficientnet_preprocess,
            'input_size': (224, 224) 
        },
        'densenet': {
            'create_function': 'create_multimodal_densenet',
            'model_name': 'MultimodalDenseNet',
            'preprocess_func': densenet_preprocess,
            'input_size': (224, 224)  
        }
    }
    return model_configs




def train_multiple_models(framework, validator, df, column_name_to_discretize, 
                         models_to_train=['resnet', 'vgg', 'efficientnet', 'densenet'], 
                         NUMBER_OF_FOLDS=1, save_dir='histories'):
    
    
    model_configs = get_model_config()
    all_histories = {}
    
    for model_type in models_to_train:
        if model_type not in model_configs:
            print(f"Modèle {model_type} non reconnu. Modèles disponibles: {list(model_configs.keys())}")
            continue
            
        config = model_configs[model_type]
        print(f"\n{'='*50}")
        print(f"ENTRAÎNEMENT DU MODÈLE: {config['model_name']}")
        print(f"{'='*50}")
        
        model_histories = []
        
        for fold_id in range(NUMBER_OF_FOLDS):
            print(f"\nFold {fold_id + 1}/{NUMBER_OF_FOLDS} pour {config['model_name']}")
            
            # Obtenir les données pour ce fold
            train_df, test_df, val_df = validator.get_fold_data(df, fold=fold_id)
            
            # Préparation des métadonnées
            X_dict, y_dict, transformers = framework.prepare_metadata(
                train_df, test_df, val_df, 'dx', column_name_to_discretize, 
                "encoders", fold_number=fold_id
            )
            
            # Prétraitement des images avec la taille appropriée pour ce modèle
            input_size = config['input_size']
            train_df["image"] = train_df["path"].map(
                lambda x: np.asarray(Image.open(x).resize(input_size))
            )
            test_df["image"] = test_df["path"].map(
                lambda x: np.asarray(Image.open(x).resize(input_size))
            )
            val_df["image"] = val_df["path"].map(
                lambda x: np.asarray(Image.open(x).resize(input_size))
            )
            
            # Préprocessing spécifique au modèle
            pre_x_train_img, pre_x_test_img, pre_x_val = framework.preprocess_image_data_with_pretrained(
                train_df, test_df, val_df, preprocess_func=config['preprocess_func']
            )
            
            # Création du modèle multimodal
            metadata_dim = X_dict["train"].shape[1]
            create_func = getattr(framework, config['create_function'])
            create_func(metadata_dim=metadata_dim)
            framework.compile_models()
            
            # Création des datasets
            train_data = make_multimodal_dataset(
                pre_x_train_img, X_dict["train"], y_dict["train"],
                batch_size=32, augment=True
            )
            val_data = make_multimodal_dataset(
                pre_x_val, X_dict['val'], y_dict['val'],
                batch_size=32, shuffle=False, augment=False
            )
            test_data = make_multimodal_dataset(
                pre_x_test_img, X_dict['test'], y_dict['test'],
                batch_size=32, shuffle=False, augment=False
            )
            
            # Entraînement du modèle
            print(f"Début de l'entraînement pour {config['model_name']} - Fold {fold_id + 1}")
            history = framework.train_model(
                model_name=config['model_name'],
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                y_true=y_dict["test"],
                fold=fold_id
            )
            
            # Sauvegarde de l'historique
            save_history_to_csv(history, fold_id, config['model_name'], save_dir=save_dir)
            model_histories.append(history)
            
            print(f"Terminé: {config['model_name']} - Fold {fold_id + 1}")
        
        all_histories[model_type] = model_histories
        print(f"\nTerminé: Tous les folds pour {config['model_name']}")
    
    return all_histories