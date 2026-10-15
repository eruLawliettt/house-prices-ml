from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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