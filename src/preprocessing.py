from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class TotalSFTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X["TotalSF"] = (
            X["TotalBsmtSF"]
            + X["1stFlrSF"]
            + X["2ndFlrSF"]
        )
        
        return X
    
class TotalBathroomsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalBathrooms"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"].fillna(0)
            + 0.5 * X["BsmtHalfBath"].fillna(0)
        )

        return X
    
class HouseAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['HouseAge'] = (
            X['YrSold'] 
            - X['YearBuilt']
        )        
        
        return X

def build_preprocessor(X, model_type='standard', use_total_sf=False, 
                       use_total_bathrooms=False, use_house_age=False):
    
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features.append("MSSubClass")
    numerical_features.remove("MSSubClass")
    
    if use_total_sf:
        numerical_features.append('TotalSF')
        
    if use_total_bathrooms:
        numerical_features.append('TotalBathrooms')
        
    if use_house_age:
        numerical_features.append('HouseAge')
    
    feature_steps = []
    
    if use_total_sf:
        feature_steps.append(
            ('total_sf', TotalSFTransformer())
        )
        
    if use_total_bathrooms:
        feature_steps.append(
            ('total_bathrooms', TotalBathroomsTransformer())
        )
        
    if use_house_age:
        feature_steps.append(
            ('house_age', HouseAgeTransformer())
        )
        
    feature_pipeline = (
        Pipeline(feature_steps)
        if feature_steps
        else 'passthrough'
    )
    
    structural_cat_missing = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]

    structural_num_missing = [
        "GarageYrBlt",
        "MasVnrArea",
    ]

    true_cat_missing = [
        "Electrical",
    ]

    true_num_missing = [
        "LotFrontage",
    ]

    regular_cat = [
        col for col in categorical_features
        if col not in structural_cat_missing
        and col not in true_cat_missing
    ]

    regular_num = [
        col for col in numerical_features
        if col not in structural_num_missing
        and col not in true_num_missing
    ]    
    
    if model_type == 'standard':
        regular_num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        
        true_num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        
        structural_num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
        ])
        
    elif model_type in ['boosting', 'catboost']:
        regular_num_transformer = SimpleImputer(strategy='median')
        true_num_transformer = SimpleImputer(strategy='median')
        structural_num_transformer = SimpleImputer(strategy='constant', fill_value=0)
        
    else: 
        raise ValueError(
            f'Unknown model_type: {model_type}'
        )
    
    if model_type == 'catboost':
        regular_cat_transformer = 'passthrough'
        true_cat_transformer = SimpleImputer(strategy='most_frequent')
        structural_cat_transformer =  SimpleImputer(strategy='constant', fill_value='None')
        
    else: 
        regular_cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        true_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        structural_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('regular_cat', regular_cat_transformer, regular_cat),
            ('true_cat_missing', true_cat_transformer, true_cat_missing),
            ('structural_cat_missing', structural_cat_transformer, structural_cat_missing),
            ('regular_num', regular_num_transformer, regular_num),
            ('true_num_missing', true_num_transformer, true_num_missing),
            ('structural_num_missing', structural_num_transformer, structural_num_missing),
        ],
        remainder='drop',
    )
    
    return Pipeline([
        ('features', feature_pipeline),
        ('columns', column_transformer),
    ])
    
    
# Only for ipynb (old approach)
def get_preprocessor(X, model_type='default'):
    
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features.append("MSSubClass")
    numerical_features.remove("MSSubClass")
    
    structural_cat_missing = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]

    structural_num_missing = [
        "GarageYrBlt",
        "MasVnrArea",
    ]

    true_cat_missing = [
        "Electrical",
    ]

    true_num_missing = [
        "LotFrontage",
    ]

    regular_cat = [
        col for col in categorical_features
        if col not in structural_cat_missing
        and col not in true_cat_missing
    ]

    regular_num = [
        col for col in numerical_features
        if col not in structural_num_missing
        and col not in true_num_missing
    ]
    
    if model_type == 'catboost':
        preprocessor = ColumnTransformer([
            ('structural_cat', SimpleImputer(strategy='constant', fill_value='None'), structural_cat_missing),
            ('true_cat', SimpleImputer(strategy='most_frequent'), true_cat_missing),
            ('regular_cat', 'passthrough', regular_cat),
            ('structural_num', SimpleImputer(strategy='constant', fill_value=0), structural_num_missing),
            ('true_num', SimpleImputer(strategy='median'), true_num_missing),
            ('regular_num', 'passthrough', regular_num)
        ])
            
        return preprocessor
    
    structural_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    true_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    structural_num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
    ])

    true_num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    regular_cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    regular_num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    
    preprocessor = ColumnTransformer([
        ('structural_cat', structural_cat_pipeline, structural_cat_missing),
        ('true_cat', true_cat_pipeline, true_cat_missing),
        ('regular_cat', regular_cat_pipeline, regular_cat),
        ('structural_num', structural_num_pipeline, structural_num_missing),
        ('true_num', true_num_pipeline, true_num_missing),
        ('regular_num', regular_num_pipeline, regular_num),
    ])
    
    return preprocessor