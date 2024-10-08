from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer


class ClassifierParameterizer(object):
    """
    Class built to utilize for model training with optuna.
    """
    CATBOOST = 'catboost'
    XGBOOST_NAME = 'xgboost'
    RF_NAME = 'random_forest'

    def __init__(self, model_name, model_parameters, seed: int = 42):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.seed = seed

    def select_model(self, **kwargs):
        """
        Select the model based on model_name.
        """
        if self.model_name == self.XGBOOST_NAME:
            return XGBClassifier(random_state=self.seed,verbose=0, **kwargs)
        if self.model_name == self.CATBOOST:
            return CatBoostClassifier(random_state=self.seed,verbose=0, **kwargs)
        elif self.model_name == self.RF_NAME:
            return RandomForestClassifier(random_state=self.seed,verbose=0, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def create_model_parameters(self, trial):
        """
        Create model-specific hyperparameters using an Optuna trial.
        This version handles float and int types with optional step parameters.
        """
        params = {}

        for param_name, param_info in self.model_parameters.items():

            if param_info['type'] == 'float':
                # Check if a step parameter exists
                if 'step' in param_info:
                    params[param_name] = trial.suggest_float(
                        name=param_name,
                        low=param_info['low'],
                        high=param_info['high'],
                        step=param_info['step'],  # Use step if present
                        log=param_info.get('log', False)
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        name=param_name,
                        low=param_info['low'],
                        high=param_info['high'],
                        log=param_info.get('log', False)  # No step, continuous search
                    )

            elif param_info['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    name=param_name,
                    low=param_info['low'],
                    high=param_info['high'],
                    step=param_info.get('step', 1)  # Use step if present, else use 1
                )

            elif param_info['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    name=param_name,
                    choices=param_info['choices']
                )

        return params
    
def objective(trial, X_train, y_train, model_name, model_parameters, categorical_features, numerical_features):
    
    model_parameterizer = ClassifierParameterizer(model_name, model_parameters)
    params = model_parameterizer.create_model_parameters(trial)
    classifier = model_parameterizer.select_model(**params)
    
    
    ### Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('classifier', classifier) 
    ])
    ###

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='macro')
    }
    
    cv_results = cross_validate(
        pipeline, 
        X_train, 
        y_train, 
        cv=kf, 
        scoring=scoring,  
        return_train_score=True,  
        n_jobs=-1
    )
    
    mean_test_accuracy = np.mean(cv_results['test_accuracy'])
    mean_test_f1 = np.mean(cv_results['test_f1'])
    
    print(f"Mean Test Accuracy: {mean_test_accuracy}")
    print(f"Mean Test F1: {mean_test_f1}")
    
    return mean_test_f1