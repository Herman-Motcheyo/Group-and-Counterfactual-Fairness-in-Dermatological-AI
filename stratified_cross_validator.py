import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import json



warnings.filterwarnings('ignore')

class StratifiedCrossValidator:
    
    def __init__(self, n_splits: int = 5, random_state: int = 42, 
                 shuffle: bool = True, min_samples_per_class: int = 10,
                 test_size: float = 0.2, val_size: float = 0.2):
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.min_samples_per_class = min_samples_per_class
        self.test_size = test_size
        self.val_size = val_size
        
        # Initialisations
        self.skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )
        self.splits_indices = None
        self.train_test_val_indices = None
        self.metadata_analysis = {}
        self.bias_warnings = []
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    metadata_cols: List[str] = None) -> pd.DataFrame:
   
        data = df.copy()
        
        # Vérifications de base
        if target_col not in data.columns:
            raise ValueError(f"Colonne cible '{target_col}' non trouvée")
        
        # Analyse de distribution des classes
        class_counts = data[target_col].value_counts()
        print(f" Distribution des classes:")
        for class_name, count in class_counts.items():
            percentage = (count / len(data)) * 100
            print(f"   {class_name}: {count} échantillons ({percentage:.1f}%)")
        
        # Vérification du nombre minimum d'échantillons par classe
        insufficient_classes = class_counts[class_counts < self.min_samples_per_class]
        if len(insufficient_classes) > 0:
            print(f"  Classes avec moins de {self.min_samples_per_class} échantillons:")
            for class_name, count in insufficient_classes.items():
                print(f"   {class_name}: {count} échantillons")
        
        return data


    def create_train_test_val_split(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Crée le split stratifié en train/test/validation.
        
        Args:
            data: DataFrame des données
            target_col: Colonne cible
            
        Returns:
            Tuple (train_data, test_data, val_data)
        """
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Premier split : séparer le test set
        if self.test_size > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y,
                shuffle=self.shuffle
            )
        else:
            X_temp, X_test, y_temp, y_test = X, pd.DataFrame(), y, pd.Series()
        
        # Deuxième split : séparer train et validation
        if self.val_size > 0 and len(X_temp) > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_temp,
                shuffle=self.shuffle
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, pd.DataFrame(), y_temp, pd.Series()
        
        # Reconstruction des DataFrames complets
        train_data = pd.concat([X_train, y_train], axis=1) if len(X_train) > 0 else pd.DataFrame()
        test_data = pd.concat([X_test, y_test], axis=1) if len(X_test) > 0 else pd.DataFrame()
        val_data = pd.concat([X_val, y_val], axis=1) if len(X_val) > 0 else pd.DataFrame()
        
        # Stockage des indices pour référence
        self.train_test_val_indices = {
            'train': train_data.index.tolist() if len(train_data) > 0 else [],
            'test': test_data.index.tolist() if len(test_data) > 0 else [],
            'val': val_data.index.tolist() if len(val_data) > 0 else []
        }
        
        # Vérification de la stratification
        self._verify_stratification(data, train_data, test_data, val_data, target_col)
        
        return train_data, test_data, val_data
    
    def _verify_stratification(self, original_data: pd.DataFrame, 
                             train_data: pd.DataFrame, test_data: pd.DataFrame, 
                             val_data: pd.DataFrame, target_col: str):
        """Vérifie que la stratification est correcte."""
        print(f"\n  Vérification de la stratification:")
        
        original_dist = original_data[target_col].value_counts(normalize=True).sort_index()
        
        for class_name in original_dist.index:
            orig_pct = original_dist[class_name] * 100
            
            # Distribution dans train
            train_pct = 0
            if len(train_data) > 0:
                train_dist = train_data[target_col].value_counts(normalize=True)
                train_pct = train_dist.get(class_name, 0) * 100
            
            # Distribution dans test
            test_pct = 0
            if len(test_data) > 0:
                test_dist = test_data[target_col].value_counts(normalize=True)
                test_pct = test_dist.get(class_name, 0) * 100
            
            # Distribution dans validation
            val_pct = 0
            if len(val_data) > 0:
                val_dist = val_data[target_col].value_counts(normalize=True)
                val_pct = val_dist.get(class_name, 0) * 100
            
    
    def _analyze_metadata_distribution(self, data: pd.DataFrame, 
                                     target_col: str, metadata_cols: List[str]):
        
        for col in metadata_cols:
            if col in data.columns:
                print(f"\n {col}:")
                
                # Distribution globale
                global_dist = data[col].value_counts(normalize=True)
                
                # Distribution par classe
                class_distributions = {}
                for class_name in data[target_col].unique():
                    class_data = data[data[target_col] == class_name]
                    class_dist = class_data[col].value_counts(normalize=True)
                    class_distributions[class_name] = class_dist
                
                # Détection des biais potentiels
                self._detect_metadata_bias(col, global_dist, class_distributions)
                
                # Stockage pour analyse ultérieure
                self.metadata_analysis[col] = {
                    'global': global_dist,
                    'by_class': class_distributions
                }


    def create_stratified_splits_finale(self, data: pd.DataFrame, target_col: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        
        X = data.index.values
        y = data[target_col].values

        splits = []

        for fold, (train_val_idx, test_idx) in enumerate(self.skf.split(X, y)):
            # Indices pour train + val
            X_train_val = X[train_val_idx]
            y_train_val = y[train_val_idx]

            # Split train/val de façon stratifiée
            train_idx, val_idx = train_test_split(
                X_train_val,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y_train_val
            )

            splits.append(
                {
                'fold': fold,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'test_indices': test_idx.tolist()
            }
            )
        self.splits_indices = splits
        with open("cross_val_index.json", "w") as f:
            json.dump(splits, f, indent=2)

        return splits

    

    
    def create_stratified_splits(self, data: pd.DataFrame, target_col: str) -> List[Tuple]:

        X = data.index.values
        y = data[target_col].values
        
        # Création des splits stratifiés
        splits = []
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            splits.append((train_idx, val_idx))
            
            # Vérification de la stratification
            train_dist = pd.Series(y[train_idx]).value_counts(normalize=True).sort_index()
            val_dist = pd.Series(y[val_idx]).value_counts(normalize=True).sort_index()
            
            print(f"\n Fold {fold + 1}:")
            print(f"   Train: {len(train_idx)} échantillons")
            print(f"   Val:   {len(val_idx)} échantillons")
            
            # Affichage des distributions
            print("   Distribution train vs val:")
            for class_name in train_dist.index:
                train_pct = train_dist[class_name] * 100
                val_pct = val_dist[class_name] * 100 if class_name in val_dist.index else 0
                print(f"     {class_name}: {train_pct:.1f}% vs {val_pct:.1f}%")
        
        self.splits_indices = splits
        return splits
    
    def validate_metadata_consistency(self, data: pd.DataFrame, 
                                    target_col: str, metadata_cols: List[str]):
        """
        Valide la cohérence des métadonnées à travers les folds.
        """
        if not self.splits_indices:
            raise ValueError("Les splits doivent être créés avant la validation")
        
        print(f"\n  Validation de la cohérence des métadonnées:")
        
        for col in metadata_cols:
            if col not in data.columns:
                continue
                
            print(f"\n {col}:")
            
            fold_distributions = []
            for fold, (train_idx, val_idx) in enumerate(self.splits_indices):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # Distribution dans le fold d'entraînement
                train_dist = train_data[col].value_counts(normalize=True)
                val_dist = val_data[col].value_counts(normalize=True)
                
                fold_distributions.append({
                    'fold': fold + 1,
                    'train': train_dist,
                    'val': val_dist
                })
            
            self._calculate_fold_variance(col, fold_distributions)
    
    def _calculate_fold_variance(self, col_name: str, fold_distributions: List[Dict]):
        """Calcule la variance des distributions entre folds."""
        categories = set()
        for fold_data in fold_distributions:
            categories.update(fold_data['train'].index)
            categories.update(fold_data['val'].index)
        
        variance_threshold = 0.05  
        
        for category in categories:
            train_proportions = []
            val_proportions = []
            
            for fold_data in fold_distributions:
                train_prop = fold_data['train'].get(category, 0)
                val_prop = fold_data['val'].get(category, 0)
                train_proportions.append(train_prop)
                val_proportions.append(val_prop)
            
            train_variance = np.var(train_proportions)
            val_variance = np.var(val_proportions)
            
            if train_variance > variance_threshold or val_variance > variance_threshold:
                print(f"     Train variance: {train_variance:.4f}")
                print(f"     Val variance: {val_variance:.4f}")
    
    def plot_train_test_val_distributions(self, train_data: pd.DataFrame, 
                                        test_data: pd.DataFrame, 
                                        val_data: pd.DataFrame, 
                                        target_col: str):
        """Visualise les distributions de classes pour train/test/validation."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Distribution des Classes: Train/Test/Validation', fontsize=16)
        
        datasets = [
            (train_data, 'Train', 'skyblue'),
            (test_data, 'Test', 'lightcoral'),
            (val_data, 'Validation', 'lightgreen')
        ]
        
        for i, (data, title, color) in enumerate(datasets):
            ax = axes[i]
            if len(data) > 0:
                dist = data[target_col].value_counts()
                ax.bar(dist.index, dist.values, color=color, alpha=0.8)
                ax.set_title(f'{title} ({len(data)} échantillons)')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Nombre d\'échantillons')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_title(f'{title} (vide)')
                ax.text(0.5, 0.5, 'Pas de données', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distributions(self, data: pd.DataFrame, target_col: str):
        """Visualise les distributions de classes par fold."""
        if not self.splits_indices:
            raise ValueError("Les splits doivent être créés avant la visualisation")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution des Classes par Fold', fontsize=16)
        
        # Distribution globale
        ax = axes[0, 0]
        global_dist = data[target_col].value_counts()
        ax.bar(global_dist.index, global_dist.values)
        ax.set_title('Distribution Globale')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Nombre d\'échantillons')
        ax.tick_params(axis='x', rotation=45)
        
        # Distributions par fold
        for fold, (train_idx, val_idx) in enumerate(self.splits_indices):
            if fold >= 5:  # Limite à 5 folds pour l'affichage
                break
                
            row = (fold + 1) // 3
            col = (fold + 1) % 3
            ax = axes[row, col]
            
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            train_dist = train_data[target_col].value_counts()
            val_dist = val_data[target_col].value_counts()
            
            x = np.arange(len(train_dist.index))
            width = 0.35
            
            ax.bar(x - width/2, train_dist.values, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, val_dist.values, width, label='Val', alpha=0.8)
            
            ax.set_title(f'Fold {fold + 1}')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Nombre d\'échantillons')
            ax.set_xticks(x)
            ax.set_xticklabels(train_dist.index, rotation=45)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    
    def get_fold_data(self, data: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if not self.splits_indices:
            raise ValueError("Les splits doivent être créés avant de récupérer les données")
        
        if fold >= len(self.splits_indices):
            raise ValueError(f"Fold {fold} non disponible (max: {len(self.splits_indices)-1})")
        
        train_idx = self.splits_indices[fold]["train_indices"]
        val_idx= self.splits_indices[fold]["val_indices"]
        test_idx = self.splits_indices[fold]["test_indices"]
        
        train_data = data.iloc[train_idx].copy()
        test_data  = data.iloc[test_idx].copy()
        val_data = data.iloc[val_idx].copy()
        
        return train_data, test_data, val_data
    
    
    def export_splits(self, output_path: str):
        """Exporte les indices des splits pour reproductibilité."""
        export_data = {
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'train_test_val_indices': self.train_test_val_indices,
            'cross_validation_splits': []
        }
        
        if self.splits_indices:
            for fold, (train_idx, val_idx) in enumerate(self.splits_indices):
                export_data['cross_validation_splits'].append({
                    'fold': fold,
                    'train_indices': train_idx.tolist(),
                    'val_indices': val_idx.tolist()
                })
        
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f" Splits exportés vers {output_path}")
    
    def load_splits(self, input_path: str):
        """Charge les indices des splits depuis un fichier."""
        import json
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.n_splits = data['n_splits']
        self.random_state = data['random_state']
        self.test_size = data.get('test_size', 0.2)
        self.val_size = data.get('val_size', 0.2)
        self.train_test_val_indices = data.get('train_test_val_indices')
        
        if 'cross_validation_splits' in data:
            self.splits_indices = []
            for split_data in data['cross_validation_splits']:
                train_idx = np.array(split_data['train_indices'])
                val_idx = np.array(split_data['val_indices'])
                self.splits_indices.append((train_idx, val_idx))
        
        print(f"Splits chargés depuis {input_path}")
    
    def get_summary(self) -> Dict:
        """Retourne un résumé des splits créés."""
        summary = {
            'configuration': {
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'test_size': self.test_size,
                'val_size': self.val_size
            },
            'train_test_val_split': self.train_test_val_indices is not None,
            'cross_validation_splits': self.splits_indices is not None,
            'bias_warnings': len(self.bias_warnings)
        }
        
        if self.train_test_val_indices:
            summary['train_test_val_sizes'] = {
                'train': len(self.train_test_val_indices['train']),
                'test': len(self.train_test_val_indices['test']),
                'val': len(self.train_test_val_indices['val'])
            }
        
        return summary