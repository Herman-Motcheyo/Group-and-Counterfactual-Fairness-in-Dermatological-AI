# Contraintes biologiques générées automatiquement
# à partir de l'analyse du dataset

lesion_constraints = {
    'bkl': {  # Benign keratosis-like lesions
        'age_range': (45, 85),
        'sex_preference': None,
        'common_locations': ['face', 'lower extremity', 'back', 'upper extremity', 'chest', 'trunk', 'abdomen']
    },
    'df': {  # Dermatofibroma
        'age_range': (35, 74),
        'sex_preference': None,
        'common_locations': ['lower extremity', 'upper extremity']
    },
    'mel': {  # Melanoma
        'age_range': (30, 85),
        'sex_preference': 'male',
        'common_locations': ['back', 'upper extremity', 'lower extremity', 'face', 'abdomen']
    },
    'vasc': {  # Vascular lesions
        'age_range': (13, 80),
        'sex_preference': None,
        'common_locations': ['abdomen', 'lower extremity', 'upper extremity', 'back', 'trunk', 'hand']
    },
    'bcc': {  # Basal cell carcinoma
        'age_range': (40, 85),
        'sex_preference': 'male',
        'common_locations': ['back', 'face', 'lower extremity', 'chest', 'upper extremity']
    },
    'nv': {  # Nevus
        'age_range': (20, 75),
        'sex_preference': None,
        'common_locations': ['lower extremity', 'back', 'trunk', 'abdomen', 'upper extremity']
    },
    'akiec': {  # Actinic keratoses
        'age_range': (41, 85),
        'sex_preference': 'male',
        'common_locations': ['face', 'lower extremity', 'upper extremity', 'hand', 'neck']
    },
}
