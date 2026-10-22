from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def get_model(model_name: str, seed: int = 42, **model_params):
    """Create and return a model by name."""
    
    if model_name == 'linear_regression':
        return LinearRegression(
            random_state=seed,
            **model_params,
        )
        
    if model_name == 'lasso':
        return Lasso(
            random_state=seed,
            **model_params,
        )
        
    if model_name == 'ridge':
        return Ridge(
            random_state=seed,
            **model_params,
        )
            
    if model_name == 'elastic_net':
        return ElasticNet(
            random_state=seed,
            **model_params,
        )
    
    if model_name == 'knn':
        return KNeighborsRegressor(
            **model_params,
        )
        
    if model_name == 'decision_tree':
        return DecisionTreeRegressor(
            random_state=seed,
            **model_params,
        )
            
    if model_name == 'random_forest':
        return RandomForestRegressor(
            random_state=seed,
            **model_params
        )
        
    if model_name == 'lightgbm':
        return LGBMRegressor(
            random_state=seed,
            verbosity=-1,
            **model_params
        )
        
    if model_name == 'xgboost':
        return XGBRegressor(
            random_state=seed,
            **model_params
        )

    if model_name == 'catboost':
        return CatBoostRegressor(
            random_seed=seed,
            verbose=False,
            **model_params
        )

    raise ValueError(f"Unknown model: {model_name}")
                
    