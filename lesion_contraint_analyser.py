import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class LesionConstraintsAnalyzer:
    def __init__(self, df, lesion_column='dx', age_column='age', 
                 sex_column='sex', location_column='localization'):
        """
        Analyseur de contraintes biologiques basé sur les données réelles
        
        Args:
            df: DataFrame contenant les données
            lesion_column: nom de la colonne des types de lésions
            age_column: nom de la colonne âge
            sex_column: nom de la colonne sexe
            location_column: nom de la colonne localisation
        """
        self.df = df
        self.lesion_col = lesion_column
        self.age_col = age_column
        self.sex_col = sex_column
        self.location_col = location_column
        self.constraints = {}
        
    def analyze_age_distribution(self, percentile_range=(5, 95)):
        age_constraints = {}
        
        for lesion_type in self.df[self.lesion_col].unique():
            lesion_data = self.df[self.df[self.lesion_col] == lesion_type]
            ages = lesion_data[self.age_col].dropna()
            
            if len(ages) < 5: 
                continue
                
            age_stats = {
                'count': len(ages),
                'mean': ages.mean(),
                'std': ages.std(),
                'median': ages.median(),
                'min': ages.min(),
                'max': ages.max(),
                'q25': ages.quantile(0.25),
                'q75': ages.quantile(0.75)
            }
            
            age_range = (
                int(ages.quantile(percentile_range[0]/100)),
                int(ages.quantile(percentile_range[1]/100))
            )
            
            age_constraints[lesion_type] = {
                'range': age_range,
                'stats': age_stats
            }
        
        return age_constraints
    
    
    
    def analyze_sex_distribution(self, significance_threshold=0.05):
        """Analyse la préférence de sexe par type de lésion
                L'objectif principal est de déterminer si la répartition observée des cas entre 
                les hommes et les femmes pour une maladie 
                donnée est le fruit du hasard ou si elle révèle une préférence épidémiologique réelle (une contrainte).
        """
        sex_constraints = {}
        
        print(f"\n ANALYSE DES PRÉFÉRENCES DE SEXE")
        print("="*50)
        
        for lesion_type in self.df[self.lesion_col].unique():
            lesion_data = self.df[self.df[self.lesion_col] == lesion_type]
            sex_counts = lesion_data[self.sex_col].value_counts()
            
            if len(sex_counts) < 2 or sex_counts.sum() < 10:
                continue
            
            # Test du chi-carré pour significativité
            total = sex_counts.sum()
            male_count = sex_counts.get('male', 0)
            female_count = sex_counts.get('female', 0)
            
            # Proportion attendue (50-50)
            expected = total / 2
            chi2_stat = ((male_count - expected)**2 + (female_count - expected)**2) / expected
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            # Déterminer la préférence
            male_ratio = male_count / total
            female_ratio = female_count / total
            
            preference = None
            if p_value < significance_threshold:  # Différence significative
                if male_ratio > female_ratio + 0.1:  # Au moins 10% de différence
                    preference = 'male'
                elif female_ratio > male_ratio + 0.1:
                    preference = 'female'
            
            sex_constraints[lesion_type] = {
                'male_count': male_count,
                'female_count': female_count,
                'male_ratio': male_ratio,
                'female_ratio': female_ratio,
                'preference': preference,
                'p_value': p_value,
                'significant': p_value < significance_threshold
            }
            
            if p_value < significance_threshold:
                print(f"  • Significatif: Oui (p={p_value:.3f})")
            else:
                print(f"  • Significatif: Non (p={p_value:.3f})")
        
        return sex_constraints
    
    def analyze_location_distribution(self, min_frequency=0.05):
        location_constraints = {}
                
        for lesion_type in self.df[self.lesion_col].unique():
            lesion_data = self.df[self.df[self.lesion_col] == lesion_type]
            location_counts = lesion_data[self.location_col].value_counts()
            
            if len(location_counts) == 0:
                continue
            
            total = location_counts.sum()
            
            # Garder seulement les localisations fréquentes
            frequent_locations = []
            location_stats = {}
            
            for location, count in location_counts.items():
                frequency = count / total
                location_stats[location] = {
                    'count': count,
                    'frequency': frequency
                }
                
                if frequency >= min_frequency:  # Au moins 5% des cas
                    frequent_locations.append(location)
            
           
            frequent_locations = sorted(frequent_locations, 
                                      key=lambda x: location_stats[x]['frequency'], 
                                      reverse=True)
    
            location_constraints[lesion_type] = {
                'common_locations': frequent_locations,
                'all_locations_stats': location_stats,
                'total_cases': total
            }
            
            print(f"\n{lesion_type.upper()} (n={total}):")
            for loc in frequent_locations[:5]:  # Top 5
                stats_loc = location_stats[loc]
                print(f"    - {loc}: {stats_loc['count']} ({stats_loc['frequency']:.1%})")
        
        return location_constraints
    
    def generate_constraints_dict(self):
        
        age_data = self.analyze_age_distribution()
        sex_data = self.analyze_sex_distribution()
        location_data = self.analyze_location_distribution()
        
        # Combiner en contraintes finales
        final_constraints = {}
        
        for lesion_type in self.df[self.lesion_col].unique():
            constraints = {}
            
            if lesion_type in age_data:
                constraints['age_range'] = age_data[lesion_type]['range']
            else:
                constraints['age_range'] = (0, 100) 
            
            # Préférence de sexe
            if lesion_type in sex_data:
                constraints['sex_preference'] = sex_data[lesion_type]['preference']
            else:
                constraints['sex_preference'] = None
            
            # Localisations communes
            if lesion_type in location_data:
                constraints['common_locations'] = location_data[lesion_type]['common_locations']
            else:
                constraints['common_locations'] = []
            
            final_constraints[lesion_type] = constraints
        
        self.constraints = final_constraints
        return final_constraints
    
    def print_constraints_code(self):
        if not self.constraints:
            self.generate_constraints_dict()
        
        
        print("="*50)
        print("self.lesion_constraints = {")
        
        for lesion_type, constraints in self.constraints.items():
            comment = self._get_lesion_comment(lesion_type)
            print(f"    '{lesion_type}': {{  {comment}")
            print(f"        'age_range': {constraints['age_range']},")
            
            if constraints['sex_preference']:
                print(f"        'sex_preference': '{constraints['sex_preference']}',")
            else:
                print(f"        'sex_preference': None,")
            
            locations = constraints['common_locations']
            print(f"        'common_locations': {locations}")
            print("    },")
        
        print("}")
    
    def _get_lesion_comment(self, lesion_type):
        comments = {
            'mel': '# Melanoma',
            'bcc': '# Basal cell carcinoma', 
            'nv': '# Nevus',
            'akiec': '# Actinic keratoses',
            'bkl': '# Benign keratosis-like lesions',
            'df': '# Dermatofibroma',
            'vasc': '# Vascular lesions'
        }
        return comments.get(lesion_type, f'# {lesion_type}')
    
    def plot_distributions(self, figsize=(15, 12)):
        """Visualise les distributions analysées"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Distribution des Caractéristiques par Type de Lésion', fontsize=16)
        
        # 1. Distribution d'âge
        lesion_types = self.df[self.lesion_col].unique()
        age_data = []
        labels = []
        
        for lesion_type in lesion_types:
            ages = self.df[self.df[self.lesion_col] == lesion_type][self.age_col].dropna()
            if len(ages) > 0:
                age_data.append(ages)
                labels.append(f"{lesion_type} (n={len(ages)})")
        
        axes[0, 0].boxplot(age_data, labels=labels)
        axes[0, 0].set_title('Distribution d\'Âge par Type de Lésion')
        axes[0, 0].set_ylabel('Âge')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribution par sexe
        sex_data = pd.crosstab(self.df[self.lesion_col], self.df[self.sex_col], normalize='index')
        sex_data.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution par Sexe')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(title='Sexe')
        
        # 3. Top localisations
        location_counts = self.df.groupby(self.lesion_col)[self.location_col].value_counts().unstack(fill_value=0)
        # Garder seulement les 8 localisations les plus fréquentes
        top_locations = location_counts.sum().nlargest(8).index
        location_counts[top_locations].plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution des Localisations')
        axes[1, 0].set_ylabel('Nombre de cas')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Localisation', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        pivot_data = self.df.pivot_table(values=self.age_col, 
                                       index=self.lesion_col, 
                                       columns=self.location_col, 
                                       aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title('Âge Moyen par Lésion et Localisation')
        
        plt.tight_layout()
        plt.show()
    
    def save_constraints_to_file(self, filename='lesion_constraints.py'):
        if not self.constraints:
            self.generate_constraints_dict()
        
        with open(filename, 'w') as f:
            f.write("# Contraintes biologiques générées automatiquement\n")
            f.write("# à partir de l'analyse du dataset\n\n")
            f.write("lesion_constraints = {\n")
            
            for lesion_type, constraints in self.constraints.items():
                comment = self._get_lesion_comment(lesion_type)
                f.write(f"    '{lesion_type}': {{  {comment}\n")
                f.write(f"        'age_range': {constraints['age_range']},\n")
                
                if constraints['sex_preference']:
                    f.write(f"        'sex_preference': '{constraints['sex_preference']}',\n")
                else:
                    f.write(f"        'sex_preference': None,\n")
                
                locations = constraints['common_locations']
                f.write(f"        'common_locations': {locations}\n")
                f.write("    },\n")
            
            f.write("}\n")
        
        print(f" Contraintes sauvegardées dans {filename}")
        
    