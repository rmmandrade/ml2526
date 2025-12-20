import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz, distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import randint,loguniform,uniform
import random
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, skew
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error,median_absolute_error
#from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
import re
from sklearn.preprocessing import PowerTransformer

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

def infer_brand_fit(df_train):

    required = {"Brand", "model"}
    if not required.issubset(df_train.columns):
        return {}
    
    model_to_brand = (
        df_train[df_train["Brand"] != "UNKNOWN"]
        .groupby("model")["Brand"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "UNKNOWN")
        .to_dict()
    )
    return model_to_brand


def infer_brand_apply(df, model_to_brand):

    if not model_to_brand:
        return df
    
    df = df.copy()
    df["Brand"] = df.apply(
        lambda row: (
            row["Brand"]
            if row["Brand"] != "UNKNOWN"
            else model_to_brand.get(row["model"], "UNKNOWN")
        ),
        axis=1
    )
    return df

def limpar_anos(df,max_year=2020):
    df = df.copy()

    df["year"] = max_year - df["year"]
    df.loc[(df["year"] < 0) | (df["year"] > 50), "year"] = np.nan #os carros de 1970 nem existiam em 1970 e os dados são referentes a 2020 carros depois disso não são válidos
    df["year"] = np.floor(df["year"]).astype("Int64")
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
    df=df.copy()
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

def mileage_per_year(df):
    age_divisor = np.maximum(df['year'],1)
    df["mileage_per_year"] = df["mileage"] / age_divisor
    return df

def power_efficiency(df):
    df["power_efficiency"] = df["engineSize"] / (df["mpg"])
    max_finite = df.loc[np.isfinite(df["power_efficiency"]), "power_efficiency"].max()
    df["power_efficiency"] = df["power_efficiency"].replace([np.inf], max_finite)
    
    return df

def drop_paint(df):
    df.drop(columns=["paintQuality%"], errors="ignore")
    return df

def price_transform(df):
    """
    Deterministic, leakage-free target transform.
    Applies log1p to price.

    This defines the modeling target and MUST be applied
    before CV and kept fixed across folds.
    """
    if "price" not in df.columns:
        return df
    
    df = df.copy()

    if (df["price"] < 0).any():
        raise ValueError("price contains negative values; log1p is invalid.")

    df["price"] = np.log1p(df["price"])

    return df

def clean_df(df, valid_models, cat_cols):
    df=df.copy()
    df=carID_como_index(df)
    df=fuzzy_marcas(df)
    df=fuzzy_modelos(df, valid_models)
    #df=inferir_marca_com_modelo(df, valid_models)
    df=limpar_anos(df)
    df=fuzzy_transmissao(df)
    df=impossible_to_nan(df, "mileage")
    df=mileage_per_year(df)
    df=fuzzy_fuel(df)
    df=impossible_to_nan(df, "tax")
    df=impossible_to_nan(df, "mpg")
    df=impossible_to_nan(df, "engineSize", 0.6)
    df=power_efficiency(df)
    df=drop_paint(df)
    df=impossible_to_nan(df,"previousOwners")
    df=round_owners_int(df)
    df=remove_hasdmg(df)
    df=fill_cats_UNKNOWN(df,cat_cols)
    df=price_transform(df)
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
    outlier_counts=[]
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
    for item in num_cols:
    #   find and count the outliers of metric features
        q1 = X[item].quantile(0.25)
        q3 = X[item].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        lower_outliers = (X[item] < lower).sum()
        upper_outliers = (X[item] > upper).sum()
        lower_pct = (lower_outliers / len(X)) * 100
        upper_pct = (upper_outliers / len(X)) * 100
        #create a dictionary with the feature,#outlier and percentage of outliers
        outlier_counts.append({
        "Feature": item,
        "Lower Outlier Count": lower_outliers,
        "Upper Outlier Count": upper_outliers,
        "Lower Outlier %": lower_pct,
        "Upper Outlier %": upper_pct
    })
    df_outlier = pd.DataFrame(outlier_counts) 
    return df_outlier

def outliers_skews_train(
    X,
    num_cols,
    *,
    q_low=0.001,
    q_high=0.999,
    upper_only_after_log=False,
):
    """
    1) For each numeric column:
    compute skewness on raw values → decide log

    2) log-transform first (if decided)

    3) compute MAD robust z-scores on working scale

    4) define clipping bounds using quantiles of the inliers

    5) apply transforms

    6) return transformed data + metadata to replay on test
    """
    X = X.copy()
    info = {}

    for col in num_cols:
        x_raw = X[col].to_numpy()

        col_info = {
            "log": False,
            "clip": False,
            "lower": None,
            "upper": None,
            "q_low": float(q_low),
            "q_high": float(q_high),
            "mad_z": float(3.5),
            "skew_raw": None,
            "upper_only": False,
        }

        #not enough data
        if len(x_raw) < 20:
            info[col] = col_info
            continue

        #log transform if skewness > 1 and non negative values
        s = float(skew(x_raw))
        col_info["skew_raw"] = s
        do_log = (s > 1) and np.all(x_raw >= 0)
        col_info["log"] = bool(do_log)
        col_info["upper_only"] = bool(do_log and upper_only_after_log)

        # scale
        x_work = np.log1p(x_raw) if do_log else x_raw

        # (3) MAD robust z-scores on working scale
        med = np.median(x_work)
        mad = np.median(np.abs(x_work - med))
        #if mad~0 the columns is mostly constant
        if mad < 1e-12:
            info[col] = col_info
            # still apply log if chosen (even if clipping is skipped)
            if do_log:
                X[col] = np.log1p(X[col])
            continue

        scale = 1.4826 * mad
        rz = (x_work - med) / scale
        inliers = np.abs(rz) <= 3.5

        # If nothing to clip, still apply log if needed and move on
        if inliers.all():
            info[col] = col_info
            if do_log:
                X[col] = np.log1p(X[col])
            continue

        # (4) Quantile bounds computed on INLIERS (still on working scale)
        x_in = x_work[inliers]
        upper = float(np.quantile(x_in, q_high))
        lower = float(np.quantile(x_in, q_low))

        col_info["clip"] = True
        col_info["upper"] = upper
        col_info["lower"] = None if col_info["upper_only"] else lower

        # (5) Apply to training data: log -> clip (bounds are on working scale)
        if do_log:
            X[col] = np.log1p(X[col])

        if col_info["upper_only"]:
            X[col] = X[col].clip(upper=upper)
        else:
            X[col] = X[col].clip(lower=lower, upper=upper)

        info[col] = col_info

    return X, info

def outliers_skews_test(X, info):
    """
    Apply the transformations learned in outliers_skews_train to test/val data.

    For each column:
      1) Apply log1p if info[col]["log"] is True
      2) Apply clipping using stored bounds (on the same scale)
    """
    X = X.copy()

    for col, col_info in info.items():
        if col not in X.columns:
            continue

        x = X[col].to_numpy()
        if np.isnan(x).any():
            raise ValueError(f"NaNs found in '{col}' before log/MAD clipping (test/val).")

        # 1) Apply log transform if training decided so
        if col_info.get("log", False):
            X[col] = np.log1p(X[col])

        # 2) Apply clipping if training decided so
        if col_info.get("clip", False):
            lower = col_info.get("lower", None)
            upper = col_info.get("upper", None)

            if lower is None and upper is not None:
                X[col] = X[col].clip(upper=upper)
            elif lower is not None and upper is not None:
                X[col] = X[col].clip(lower=lower, upper=upper)
            elif lower is not None:
                X[col] = X[col].clip(lower=lower)

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

def run_model(X, y, int_cols, float_cols, model_class, model_params=None,
              scaler=None, encoder=None, cat_cols=None, encoding_type="ohe"):

    X_processed = X.copy()

    if model_params is None:
        model_params = {}

    X_processed, fill_values = fill_nans(X_processed, int_cols, float_cols)
    X_processed, outlier_info = outliers_skews_train(X_processed)

    freq_values = None
    train_feature_cols = None
    freq_encoded_cols = []  # <-- track freq-encoded column names

    if cat_cols is not None and len(cat_cols) > 0:
        if encoding_type == "ohe" and encoder is not None:
            encoder.fit(X_processed[cat_cols])
            encoded_array = encoder.transform(X_processed[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)

            X_processed = pd.concat([
                X_processed.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(encoded_array, columns=encoded_cols).reset_index(drop=True)
            ], axis=1)

        elif encoding_type == "freq":
            X_processed, freq_values = frequency_encode_train(X_processed, cat_cols)
            freq_encoded_cols = list(cat_cols)  # assuming freq encoding overwrites these cols

    train_feature_cols = X_processed.columns.tolist()

    if scaler is not None:
        # base numeric cols
        numeric_cols = [c for c in X_processed.columns if c in (int_cols + float_cols)]

        # if freq-encoding, also scale those (they're now numeric)
        if encoding_type == "freq":
            numeric_cols = list(dict.fromkeys(numeric_cols + freq_encoded_cols))  # de-dupe, keep order

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

        # also scale freq-encoded categorical cols (now numeric)
        if encoding_type == "freq" and cat_cols is not None:
            numeric_cols = list(dict.fromkeys(numeric_cols + list(cat_cols)))  # de-dupe

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
            scaler=fitted_scaler, encoder=fitted_encoder, cat_cols=cat_cols, train_feature_cols=train_feature_cols,
            freq_values=freq_values, outlier_info=outlier_info, encoding_type=encoding_type, return_predictions=True
        )

        y_val_pred = evaluate_model(
            X_val, y_val, trained_model, int_cols, float_cols, fill_values,
            scaler=fitted_scaler, encoder=fitted_encoder, cat_cols=cat_cols, train_feature_cols=train_feature_cols,
            freq_values=freq_values, outlier_info=outlier_info, encoding_type=encoding_type, return_predictions=True
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
        if encoding_type == "ohe" and encoder is not None:
            encoded_array = encoder.transform(X_processed[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)

            X_processed = pd.concat([
                X_processed.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(encoded_array, columns=encoded_cols).reset_index(drop=True)
            ], axis=1)

            # Align to full training feature list
            X_processed = X_processed.reindex(columns=train_feature_cols, fill_value=0)

        elif encoding_type == "freq":
            X_processed = frequency_encode_test(X_processed, cat_cols, freq_values)

    # Scale features
    if scaler is not None:
        numeric_cols = [col for col in X_processed.columns if col in (int_cols + float_cols)]

        # also scale freq-encoded categorical cols (now numeric)
        if encoding_type == "freq" and cat_cols is not None:
            numeric_cols = list(dict.fromkeys(numeric_cols + list(cat_cols)))  # de-dupe

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
                    if k not in ["mean_mae", "std_mae", "mean_r2", "std_r2"]}      # sem base_params

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
