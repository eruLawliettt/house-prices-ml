from sklearn.model_selection import KFold, cross_val_score

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