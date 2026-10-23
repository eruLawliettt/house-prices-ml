from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error
from src.models import get_model
from src.preprocessing import build_preprocessor
import numpy as np


def evaluate_model(estimator, X, y, n_splits=5, 
                   seed=42, n_jobs=1):
    """Evaluate an estimator using cross-validation."""
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
    )
    
    rmse = -scores
    
    return {
        'fold_rmse': rmse,
        'mean': rmse.mean(),
        'std': rmse.std(),
    }
    
def evaluate_catboost(X, y, model_params=None, n_splits=5, seed=42, early_stopping_rounds=None,
                      use_total_sf=False, use_total_bathrooms=False, use_house_age=False):
    
    if model_params is None:
        model_params = {}
        
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    fold_rmse = []
    best_iterations = []
    
    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type="catboost",
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        n_cat_features = (
            len(
                X_train.select_dtypes(
                    include=["object", "string"]
                ).columns
            )
            + 1  # MSSubClass
        )
        
        cat_features = list(range(n_cat_features))
        
        model = get_model(
            'catboost',
            seed=seed,
            **model_params
        )
        
        model.fit(
            X_train_processed,
            y_train,
            cat_features=cat_features,
            eval_set=(X_val_processed, y_val),
            early_stopping_rounds=early_stopping_rounds,
        )
        
        best_iterations.append(model.get_best_iteration())
        
        preds = model.predict(X_val_processed)
        
        rmse = root_mean_squared_error(y_val, preds)
        
        fold_rmse.append(rmse)
        
    fold_rmse = np.array(fold_rmse)
    best_iterations = np.array(best_iterations)
        
    return {
        "fold_rmse": fold_rmse,
        "mean_rmse": fold_rmse.mean(),
        "std_rmse": fold_rmse.std(),
        "mean_best_iteration": best_iterations,
    }