from omegaconf import OmegaConf

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.evaluation import evaluate_model, evaluate_catboost, evaluate_dnn


def main():
    
    config = OmegaConf.load('config/config.yaml')
    
    train = pd.read_csv(config.paths.train_path)
        
    target = config.target.column

    X = train.drop(columns=target)
    y = train[target]
    
    if config.target.log_transform:
        y = np.log1p(y)
        
    results = []
    
    for model_name, model_cfg in config.models.items():
        if not model_cfg.enabled:
            continue
        
        print(f'running: {model_name}')
        
        result = evaluate_configured_model(
            model_name=model_name,
            model_cfg=model_cfg,
            X=X,
            y=y,
            config=config
        )
        
        results.append({
            "model": model_name,
            "cv_mean": result["mean_rmse"],
            "cv_std": result["std_rmse"],
        })
        
    if not results:
        raise ValueError("No models are enabled in config.")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values(
        "cv_mean",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        config.paths.results_path,
        index=False,
    )
    
    print('\nResults:')
    print(results_df)
    
    best_model_name = results_df.iloc[0]["model"]
    best_score = results_df.iloc[0]["cv_mean"]

    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")

        
def evaluate_configured_model(model_name, model_cfg, X, y, config):
    
    model_params = OmegaConf.to_container(model_cfg.params, resolve=True)
    
    if model_name == 'dnn':
        result = evaluate_dnn(
            X=X,
            y=y,
            model_class=THpMLP,
            model_params=model_params,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            epochs=model_cfg.training.epochs,
            batch_size=model_cfg.training.batch_size,
            lr=model_cfg.training.lr,
            weight_decay=model_cfg.training.weight_decay,
            
            use_house_age=config.preprocessing.use_house_age,
            use_total_bathrooms=config.preprocessing.use_total_bathrooms,
            use_total_sf=config.preprocessing.use_total_sf,
        )
        
        return result
    
    if model_name == 'catboost':
        result = evaluate_catboost(
            X=X,
            y=y,
            model_params=model_params,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            early_stopping_rounds=model_cfg.training.early_stopping_rounds,
            
            use_house_age=config.preprocessing.use_house_age,
            use_total_bathrooms=config.preprocessing.use_total_bathrooms,
            use_total_sf=config.preprocessing.use_total_sf,
        )
        
        return result
    
    preprocessor = build_preprocessor(
        X=X,
        model_type=model_cfg.model_type, 
        use_total_sf=config.preprocessing.use_total_sf, 
        use_total_bathrooms=config.preprocessing.use_total_bathrooms, 
        use_house_age=config.preprocessing.use_house_age
        )
    
    model = get_model(
        model_name=model_name,
        seed=config.general.seed,
        **model_params
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return evaluate_model(
        estimator=pipeline,
        X=X,
        y=y,
        n_splits=config.cv.n_splits,
        scoring=config.cv.scoring,
        seed=config.general.seed,
    )
      
class THpMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    
if __name__ == "__main__":
    main()