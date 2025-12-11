# %%
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz, distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
#from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

#RANDOM_SEED = 1907
#random.seed(RANDOM_SEED)
#np.random.seed(RANDOM_SEED)

def carID_como_index(df):
    df = df.copy()
    df = df.set_index("carID")
    return df

def HELPER_marca_correta(marca, marcas_dict, threshold):
    if not isinstance(marca, str) or marca.strip() == "":
        return marca

    marca_lower = marca.lower().strip()

    result = process.extractOne(
        marca_lower,
        marcas_dict.keys(),
        scorer=fuzz.token_sort_ratio
    )
    
    #result = (match, score, index)
    #ex: match = toyota, score = 94.5, index = 1 (qual das marcas no dicionário é mais semelhante)

    if result is None:
        return marca

    match_lower, score, _ = result
    return marcas_dict[match_lower] if score >= threshold else marca

def fuzzy_marcas(df, threshold=50):
    df=df.copy()

    valid_brands = ["VW", "Toyota", "Audi", "Ford", "BMW", "Opel", "Skoda", "Mercedes", "Hyundai"]
    brands_dict = {BRAND.lower(): BRAND for BRAND in valid_brands}

    df["Brand"] = df["Brand"].apply(lambda x: HELPER_marca_correta(x, brands_dict, threshold))
    return df

"""Optained through meticulous ChatGPT prompt engineering"""
valid_models = {
    "Audi": [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "Q2", "Q3", "Q5", "Q7", "Q8",
        "TT", "T", "R8",
        "S3", "S4", "S5", "S8",
        "RS3", "RS4", "RS5", "RS6",
        "SQ5", "SQ7"
    ],

    "Ford": [
        "FOCUS", "FIESTA", "MONDEO", "KA", "KA+", "FUSION",
        "KUGA", "ECOSPORT", "EDGE", "PUMA",
        "CMAX", "BMAX", "SMAX", "GALAXY",
        "GRANDCMAX", "TOURNEOCONNECT", "GRANDTOURNEOCONNECT", "TOURNEOCUSTOM",
        "MUSTANG", "RANGER", "ESCORT", "STREETKA"
    ],

    "Mercedes": [
        "ACLASS", "BCLASS", "CCLASS", "ECLASS", "SCLASS",
        "GLA", "GLB", "GLC", "GLE", "GLS", "GCLASS", "GLCLASS", "MCLASS",
        "CLA", "CLS", "SL", "SLK", "CLK",
        "VCLASS", "XCLASS", "CLC"
    ],

    "VW": [
        "GOLF", "POLO", "PASSAT", "JETTA", "ARTEON", "SCIROCCO", "BEETLE",
        "UP", "GOL", "FOX",
        "TIGUAN", "TIGUANALLSPACE", "TROC", "TCROSS", "TOUAREG",
        "TOURAN", "SHARAN", "CADDY", "CADDYMAXI", "CADDYMAXILIFE",
        "CARAVELLE", "CALIFORNIA", "SHUTTLE",
        "AMAROK", "GOLFSV", "CC"
    ],

    "Opel": [
        "CORSA", "ASTRA", "INSIGNIA", "VECTRA",
        "MOKKA", "MOKKAX", "CROSSLAND", "CROSSLANDX",
        "GRANDLAND", "GRANDLANDX", "ANTARA",
        "ZAFIRA", "ZAFIRATOURER", "MERIVA", "COMBOLIFE", "VIVARO",
        "ADAM", "AGILA", "VIVA",
        "TIGRA", "GTC", "CASCADA", "AMPERA"
    ],

    "BMW": [
        "1SERIES", "2SERIES", "3SERIES", "4SERIES",
        "5SERIES", "6SERIES", "7SERIES", "8SERIES",
        "X1", "X2", "X3", "X4", "X5", "X6", "X7",
        "M2", "M3", "M4", "M5", "M6",
        "Z3", "Z4",
        "I3", "I4", "I8"
    ],

    "Toyota": [
        "YARIS", "AYGO", "AURIS", "COROLLA", "AVENSIS", "CAMRY", "PRIUS",
        "CHR", "RAV4", "LANDCRUISER", "URBANCRUISER",
        "VERSO", "VERSOS", "PROACEVERSO",
        "HILUX", "GT86", "SUPRA", "IQ"
    ],

    "Skoda": [
        "FABIA", "OCTAVIA", "SUPERB", "RAPID", "SCALA",
        "KODIAQ", "KAROQ", "KAMIQ", "YETI", "YETIOUTDOOR",
        "CITIGO", "ROOMSTER"
    ],

    "Hyundai": [
        "I10", "I20", "I30", "I40", "ACCENT", "GETZ",
        "KONA", "TUCSON", "SANTAFE", "IX20", "IX35",
        "I800", "IONIQ", "VELOSTER", "TERRACAN"
    ]
}

def HELPER_normalize_models(df):
    df = df.copy()
    
    df["model"] = (
        df["model"]
        .astype(str)
        .str.upper()
        .str.replace("-", "", regex=False)       # remove hyphens
        .str.replace(r"\s+", "", regex=True)     # remove all whitespace
        .replace(["", "NAN", "NONE"], None)
    )
    return df

def HELPER_hybrid_scorer(a, b, **kwargs):
    #hibrido entre o método tokenizer e o levenshtein 
    lev = distance.Levenshtein.normalized_similarity(a, b)
    token = fuzz.token_sort_ratio(a, b) / 100
    return (0.7 * lev + 0.3 * token) * 100

def HELPER_modelo_correto(model, brand, valid_models_dict, threshold):
    if not model or model.strip() == "":
        return None

    if not brand:
        return model
    
    # Skip unknown brands
    if brand not in valid_models_dict:
        return model
    
    valid_list = valid_models_dict[brand]

    result = process.extractOne(model, valid_list, scorer=HELPER_hybrid_scorer)

    if result is None:
        return model    

    model_name, score, _ = result
    return model_name if score >= threshold else None

def fuzzy_modelos(df, valid_models_dict, threshold=30):
    df = HELPER_normalize_models(df)
    
    df["model"] = df.apply(
        lambda row: HELPER_modelo_correto(row["model"], row["Brand"], valid_models_dict, threshold),
        axis=1
    )
    return df

def inferir_marca_com_modelo(df, valid_models_dict):
    df = df.copy()

 # Build a mapping of models → their most frequent (mode) brand in the dataset
    model_to_brand = (
        df[df["Brand"].notna()]
        .groupby("model")["Brand"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .to_dict()
    )

    df["Brand"] = df.apply(
    lambda row: (
        row["Brand"]
        if pd.notna(row["Brand"])
        else model_to_brand.get(row["model"], None)
    ),
    axis=1
    )

    return df

def limpar_anos(df):
    df = df.copy()

    df["year"] = np.round(df["year"]).astype("float")
    df.loc[(df["year"] < 1980) | (df["year"] > 2020),"year"] = np.nan
    df["year"] = df["year"].astype("Int64")
    return df

def preco_no_fim(df):
    df = df.copy()

    df = df[[col for col in df.columns if col != "price"] + ["price"]]
    return df

def HELPER_normalize_transmission(df):
    df = df.copy()
    df["transmission"] = (
        df["transmission"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(["", "NAN", "NONE"], np.nan)
    )
    return df

def HELPER_transmissao_correta(transm, valid_list, threshold):
    if pd.isna(transm):
        return np.nan

    if transm in valid_list:
        return transm
    
    result = process.extractOne(transm, valid_list, scorer=fuzz.token_sort_ratio)
    if result is None:
        return np.nan

    match_name, score, _ = result
    return match_name if score >= threshold else np.nan

def fuzzy_transmissao(df, threshold=60):
    df = HELPER_normalize_transmission(df)
    valid_list = ["MANUAL", "AUTOMATIC", "SEMI-AUTO", "OTHER", "UNKNOWN"]

    df["transmission"] = df["transmission"].apply(
        lambda x: HELPER_transmissao_correta(x, valid_list, threshold)
    )
    return df

def impossible_to_nan(df, col, val=0, lower_upper="lower"):
    df=df.copy()

    if lower_upper=="lower":
        df.loc[df[col]<val, col] = np.nan
        return df
    else:
        df.loc[df[col]>val, col] = np.nan
        return df

def HELPER_normalize_fueltype(df):
    df = df.copy()
    df["fuelType"] = (
        df["fuelType"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(["", "NAN", "NONE"], np.nan)
    )
    return df

def HELPER_fuel_correto(fuel, valid_list, threshold):
    if pd.isna(fuel):
        return np.nan

    if fuel in valid_list:
        return fuel
    
    result = process.extractOne(fuel, valid_list, scorer=fuzz.token_sort_ratio)
    if result is None:
        return np.nan

    match_name, score, _ = result
    return match_name if score >= threshold else np.nan

def fuzzy_fuel(df, threshold=60):
    df = HELPER_normalize_fueltype(df)
    valid_list = ["PETROL", "DIESEL", "HYBRID", "OTHER", "ELECTRIC"]

    df["fuelType"] = df["fuelType"].apply(
        lambda x: HELPER_fuel_correto(x, valid_list, threshold)
    )
    return df

def impossible_to_nan(df, col, val=0, lower_upper="lower"):
    df=df.copy()

    if lower_upper=="lower":
        df.loc[df[col]<val, col] = np.nan
        return df
    else:
        df.loc[df[col]>val, col] = np.nan
        return df

def round_owners_int(df):
    df=df.copy()
    df["previousOwners"] = pd.to_numeric(df["previousOwners"], errors="coerce")

    df['previousOwners'] = df['previousOwners'].round().astype('Int64')
    return df

def remove_hasdmg(df):
    df = df.copy()

    df = df.drop(columns=['hasDamage'])
    return df

def fill_cats_UNKNOWN(df, cats):
    df = df.copy()
    for column in cats:
        df[column] = df[column].fillna('UNKNOWN')
    
    return df

def clean_df(df, valid_models, cat_cols):
    df=df.copy()

    df=carID_como_index(df)
    df=fuzzy_marcas(df)
    df=fuzzy_modelos(df, valid_models)
    df=inferir_marca_com_modelo(df, valid_models)
    df=limpar_anos(df)
    df=fuzzy_transmissao(df)
    df=impossible_to_nan(df, "mileage")
    df=fuzzy_fuel(df)
    df=impossible_to_nan(df, "tax")
    df=impossible_to_nan(df, "mpg")
    df=impossible_to_nan(df, "engineSize", 0.49)
    df=impossible_to_nan(df,"paintQuality%", 100, "upper")
    df=impossible_to_nan(df,"previousOwners")
    df=round_owners_int(df)
    df=remove_hasdmg(df)
    df=fill_cats_UNKNOWN(df,cat_cols)
    
    return df

def separar_y(df):
    df=df.copy()

    X = df.drop('price', axis = 1)
    y = df['price']

    return X, y

def fill_nans(X, ints, floats, fill_values=None):
    X = X.copy()

    if fill_values is None:
        fill_values = {"float": {}, "int": {}}
        for column in floats:
            mean_to_fill = X[column].mean()
            X[column] = X[column].fillna(mean_to_fill)
            fill_values["float"][column] = mean_to_fill
    
        for column in ints:
            median_to_fill = X[column].median()
            X[column] = X[column].fillna(median_to_fill).astype("Int64")
            fill_values["int"][column] = median_to_fill

        return X, fill_values
    else:
        for col in floats:
            X[col] = X[col].fillna(fill_values["float"][col])
        for col in ints:
            X[col] = X[col].fillna(fill_values["int"][col]).astype("Int64")

        return X

def plot_nums(X, num_cols):
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(col, fontsize=14, fontweight='bold')

        # Boxplot
        sns.boxplot(y=X[col], ax=axes[0], color='skyblue')
        axes[0].set_title("Boxplot")

        # Histogram
        sns.histplot(X[col], kde=True, ax=axes[1], color='salmon')
        axes[1].set_title("Histogram")

        plt.tight_layout()
        plt.show()

def outliers_skews_train(X):
    X = X.copy()
    outlier_info = {}

    if "mileage" in X.columns:
        X["mileage"] = np.log1p(X["mileage"])
        outlier_info["mileage"] = {"log_transform": True}

    if "tax" in X.columns:
        upper = X["tax"].quantile(0.975)
        X["tax"] = X["tax"].clip(upper=upper)
        outlier_info["tax"] = {"upper": upper}

    if "mpg" in X.columns:
        upper = X["mpg"].quantile(0.975)
        X["mpg"] = X["mpg"].clip(upper=upper)
        outlier_info["mpg"] = {"upper": upper}

    if "engineSize" in X.columns:
        lower = X["engineSize"].quantile(0.01)
        upper = X["engineSize"].quantile(0.99)
        X["engineSize"] = X["engineSize"].clip(lower=lower, upper=upper)
        outlier_info["engineSize"] = {"lower": lower, "upper": upper}

    return X, outlier_info

def outliers_skews_test(X, outlier_info):
    X = X.copy()

    if "mileage" in outlier_info and "mileage" in X.columns:
        if outlier_info["mileage"].get("log_transform", False):
            X["mileage"] = np.log1p(X["mileage"])

    if "tax" in outlier_info and "tax" in X.columns:
        X["tax"] = X["tax"].clip(upper=outlier_info["tax"]["upper"])

    if "mpg" in outlier_info and "mpg" in X.columns:
        X["mpg"] = X["mpg"].clip(upper=outlier_info["mpg"]["upper"])
        
    if "engineSize" in outlier_info and "engineSize" in X.columns:
        X["engineSize"] = X["engineSize"].clip(
            lower=outlier_info["engineSize"]["lower"],
            upper=outlier_info["engineSize"]["upper"]
        )

    return X

def cor_heatmap(cor):
    plt.figure(figsize=(12,10))
    sns.heatmap(data = cor, annot = True, cmap = plt.cm.Reds, fmt='.1')
    plt.show()


def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)

def cramers_v(x, y, var):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    result =  np.sqrt(chi2 / (n * (min(k - 1, r - 1))))
    print(f"{var}: {result}")

def mutual_info(X, y, var):
    le = LabelEncoder()
    X_encoded = le.fit_transform(X.astype(str)).reshape(-1, 1)

    mi = mutual_info_classif(X_encoded, y, discrete_features=True, random_state=0)
    print(f"{var}: {mi[0]:.4f}")

def anova(X_col, y, var):
    groups = [y[X_col == level] for level in X_col.dropna().unique()]

    # Perform one-way ANOVA
    f_stat, p_val = f_oneway(*groups)

    print(f"{var.upper()}")
    print(f"  F-stat:  {f_stat:.3f}")
    print(f"  p-value: {p_val:.6f}")
    print("\n")  # line spacing between variables

    return pd.Series({"F-stat": f_stat, "p-value": p_val}, name=var)

def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()

def frequency_encode_train(df, cat_cols):
    df_encoded = df.copy()
    
    freq_values = {}
    for col in cat_cols:
        freq_map = df[col].value_counts(normalize=True)

        #this needed in order to not have data leakage when applying to test data, there will be classes that are not in the training dataset
        #due to the fact that the strings were corrupted and pre processing isnt perfect of course
        mean_freq = freq_map.mean()
        df_encoded[col] = df[col].map(freq_map).fillna(mean_freq)
        freq_values[col] = {
            "map": freq_map,
            "mean": mean_freq
        }

    return df_encoded, freq_values

def frequency_encode_test(df, cat_cols, freq_values):
    df_encoded = df.copy()

    for col in cat_cols:
        freq_map = freq_values[col]["map"]
        mean_freq = freq_values[col]["mean"]
        df_encoded[col] = df[col].map(freq_map).fillna(mean_freq)
    return df_encoded

def run_model(X, y, int_cols, float_cols, model_class, model_params=None, scaler=None, encoder=None, cat_cols=None, encoding_type="ohe"):
    X_processed = X.copy()

    if model_params is None:
        model_params = {}

    X_processed, fill_values = fill_nans(X_processed, int_cols, float_cols)
    X_processed, outlier_info = outliers_skews_train(X_processed)

    freq_values = None
    train_feature_cols = None

    if cat_cols is not None and len(cat_cols) > 0:
        if encoding_type == "ohe" and encoder is not None:
            encoder.fit(X_processed[cat_cols])
            encoded_array = encoder.transform(X_processed[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)

            X_encoded = pd.concat([
                X_processed.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(encoded_array, columns=encoded_cols).reset_index(drop=True)
            ], axis=1)
            X_processed = X_encoded

        elif encoding_type == "freq":
            X_processed, freq_values = frequency_encode_train(X_processed, cat_cols)

    train_feature_cols = X_processed.columns.tolist()

    if scaler is not None:
        numeric_cols = [col for col in X_processed.columns if col in (int_cols + float_cols)]
        X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])

    model = model_class(**model_params)
    model.fit(X_processed, y)

    return model, scaler, encoder, fill_values, train_feature_cols, freq_values, outlier_info

def evaluate_model(X, y, model, int_cols, float_cols, fill_values,
                   scaler=None, encoder=None, cat_cols=None,
                   train_feature_cols=None, freq_values=None, outlier_info=None,
                   encoding_type="ohe", return_predictions=False):
    X_processed = X.copy()

    X_processed = fill_nans(X_processed, int_cols, float_cols, fill_values)

    if outlier_info is not None:
        X_processed = outliers_skews_test(X_processed, outlier_info)

    if cat_cols is not None and len(cat_cols) > 0:
        if encoding_type == "ohe" and encoder is not None:
            encoded_array = encoder.transform(X_processed[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)

            X_encoded = pd.concat([
                X_processed.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(encoded_array, columns=encoded_cols).reset_index(drop=True)
            ], axis=1)

            # Align to training columns
            X_encoded = X_encoded.reindex(columns=train_feature_cols, fill_value=0)
            X_processed = X_encoded
            
        elif encoding_type == "freq":
            X_processed = frequency_encode_test(X_processed, cat_cols, freq_values)

    if scaler is not None:
        numeric_cols = [col for col in X_processed.columns if col in (int_cols + float_cols)]
        X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])

    preds = model.predict(X_processed)
    if return_predictions:
        return preds
    else:
        return model.score(X_processed, y)

def avg_score(method, X, y, int_cols, float_cols, model_class, model_params=None,
              scaler=None, encoder=None, cat_cols=None, encoding_type="ohe"):
    
    if model_params is None:
        model_params = {}
    
    mae_train, mae_val = [], []
    r2_train, r2_val = [], []

    for train_index, val_index in method.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        trained_model, fitted_scaler, fitted_encoder, fill_values, train_feature_cols, freq_values, outlier_info = run_model(
            X_train, y_train,int_cols, float_cols, model_class=model_class, model_params=model_params,
            scaler=scaler, encoder=encoder,cat_cols=cat_cols, encoding_type=encoding_type)

        y_train_pred = evaluate_model(
            X_train, y_train, trained_model, int_cols, float_cols, fill_values,
            fitted_scaler, fitted_encoder, cat_cols, train_feature_cols,
            freq_values, outlier_info, encoding_type, return_predictions=True
        )

        y_val_pred = evaluate_model(
            X_val, y_val, trained_model, int_cols, float_cols, fill_values,
            fitted_scaler, fitted_encoder, cat_cols, train_feature_cols,
            freq_values, outlier_info, encoding_type, return_predictions=True
        )

        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_val.append(mean_absolute_error(y_val, y_val_pred))
        r2_train.append(r2_score(y_train, y_train_pred))
        r2_val.append(r2_score(y_val, y_val_pred))

    final_model, fitted_scaler, fitted_encoder, fill_values, train_feature_cols, freq_values, outlier_info = run_model(
        X, y, int_cols, float_cols, model_class=model_class, model_params=model_params, scaler=scaler, encoder=encoder,
        cat_cols=cat_cols, encoding_type=encoding_type)

    print(f"Average Train MAE: {np.mean(mae_train):.4f}")
    print(f"Average Val MAE:   {np.mean(mae_val):.4f}")
    print(f"Average Train R²:  {np.mean(r2_train):.4f}")
    print(f"Average Val R²:    {np.mean(r2_val):.4f}")

    return {
        "mae_train": mae_train,
        "mae_val": mae_val,
        "r2_train": r2_train,
        "r2_val": r2_val,
        "final_model": final_model,
        "fitted_scaler": fitted_scaler,
        "fitted_encoder": fitted_encoder,
        "fill_values": fill_values,
        "train_feature_cols": train_feature_cols,
        "freq_values": freq_values,
        "outlier_info": outlier_info
    }

def predict_test(X_test, model, int_cols, float_cols, fill_values,
                 scaler=None, encoder=None, cat_cols=None,
                 train_feature_cols=None, freq_values=None,
                 outlier_info=None, encoding_type="ohe"):
    X_processed = X_test.copy()

    # Fill missing values
    X_processed = fill_nans(X_processed, int_cols, float_cols, fill_values)

    # Handle outliers / skews
    if outlier_info is not None:
        X_processed = outliers_skews_test(X_processed, outlier_info)

    # Encode categorical features
    if cat_cols is not None and len(cat_cols) > 0:
        if encoding_type=="ohe" and encoder is not None:
            encoded_array = encoder.transform(X_processed[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)

            X_encoded = pd.concat([
                X_processed.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(encoded_array, columns=encoded_cols).reset_index(drop=True)
            ], axis=1)

            # Align to full training feature list
            X_encoded = X_encoded.reindex(columns=train_feature_cols, fill_value=0)
            X_processed = X_encoded

        elif encoding_type == "freq":
            X_processed = frequency_encode_test(X_processed, cat_cols, freq_values)


    # Scale features
    if scaler is not None:
        numeric_cols = [col for col in X_processed.columns if col in (int_cols + float_cols)]
        X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])
    
    return model.predict(X_processed)

def random_search_cv(
    X, y,
    int_cols, float_cols, cat_cols,
    model_class=RandomForestRegressor,
    base_params=None, # fixed params, e.g. {"random_state": 42, "n_jobs": -1}
    param_dist=None,
    n_iter=30,
    method=None,
    scaler=None,
    encoder=None,
    encoding_type="ohe",
    random_seed=42
):

    if param_dist is None:
        param_dist = {
            "n_estimators": [50, 75, 100, 150, 200, 300, 400, 500, 600],
            "max_depth": [5, 7, 10, 15, 20, 25, 30],
            "min_samples_split": [2, 4, 5, 10, 12, 15],
            "min_samples_leaf": [1, 2, 5, 7, 10],
            "max_features": ["sqrt", "log2", 0.8, 0.5, 0.7],
            "max_samples": [0.7, 0.8, 0.9],
            "bootstrap": [True]
        }

    #Randomly sample n_iter parameter combinations
    param_samples = [
        {key: random.choice(values) for key, values in param_dist.items()}
        for _ in range(n_iter)
    ]

    #Default to 7-fold CV if not specified
    if method is None:
        method = KFold(n_splits=7, shuffle=True, random_state=random_seed)

    cv_search_results = []

    print(f"\nRandomized Search with {method.get_n_splits()}-Fold CV (MAE Optimization)")
    print(f"Testing {n_iter} random parameter combinations\n")

    for i, sampled_params in enumerate(param_samples, start=1):
        fold_mae_train, fold_r2_train = [], []
        fold_mae_val, fold_r2_val = [], []

        base_params = base_params or {}
        model_params = {**base_params, **sampled_params}

        print(f"► Combination {i}/{n_iter}: {model_params}")

        for fold, (train_idx, val_idx) in enumerate(method.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


            trained_model, fitted_scaler, fitted_encoder, fill_values, train_feature_cols, freq_values, outlier_info = run_model(
                X_train, y_train, int_cols, float_cols, model_class=model_class, model_params=model_params,
                scaler=scaler, encoder=encoder, cat_cols=cat_cols, encoding_type=encoding_type)

            print(f"[DEBUG] Combo {i}, Fold {fold}, model id = {id(trained_model)}")

            y_train_pred = evaluate_model(
                X_train, y_train, trained_model, int_cols, float_cols, fill_values,
                fitted_scaler, fitted_encoder, cat_cols, train_feature_cols,
                freq_values, outlier_info, encoding_type, return_predictions=True
            )

            y_val_pred = evaluate_model(
                X_val, y_val, trained_model, int_cols, float_cols, fill_values,
                fitted_scaler, fitted_encoder, cat_cols, train_feature_cols,
                freq_values, outlier_info, encoding_type, return_predictions=True
            )

            # Compute metrics
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_val = mean_absolute_error(y_val, y_val_pred)
            r2_train = r2_score(y_train, y_train_pred)
            r2_val = r2_score(y_val, y_val_pred)

            fold_mae_train.append(mae_train)
            fold_mae_val.append(mae_val)
            fold_r2_train.append(r2_train)
            fold_r2_val.append(r2_val)

            print(f"   Fold {fold}: Train MAE = {mae_train:.4f}, Val MAE = {mae_val:.4f}, "
                  f"Train R² = {r2_train:.4f}, Val R² = {r2_val:.4f}")

        mean_mae_train = np.mean(fold_mae_train)
        mean_mae_val = np.mean(fold_mae_val)
        mean_r2_train = np.mean(fold_r2_train)
        mean_r2_val = np.mean(fold_r2_val)

        cv_search_results.append({
            **model_params,
            "mean_mae_train": mean_mae_train,
            "mean_mae_val": mean_mae_val,
            "mean_r2_train": mean_r2_train,
            "mean_r2_val": mean_r2_val
        })

        print(f"→ Avg Train MAE: {mean_mae_train:.4f} | Avg Val MAE: {mean_mae_val:.4f} | "
              f"Avg Train R²: {mean_r2_train:.4f} | Avg Val R²: {mean_r2_val:.4f}\n")

    #best parameter combo
    best_result = min(cv_search_results, key=lambda x: x["mean_mae_val"])
    best_params = {k: v for k, v in best_result.items()
                    if k in param_dist.keys()}      # sem base_params

    print("\nBest Parameters (based on lowest validation MAE):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest Train MAE: {best_result['mean_mae_train']:.4f}")
    print(f"Best Val MAE:   {best_result['mean_mae_val']:.4f}")
    print(f"Best Train R²:  {best_result['mean_r2_train']:.4f}")
    print(f"Best Val R²:    {best_result['mean_r2_val']:.4f}")

    return {
        "cv_results": cv_search_results,
        "best_result": best_result,
        "best_params": best_params
    }


