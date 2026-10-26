import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import sqlite3
import threading
from contextlib import closing
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import requests
import pycountry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import (
    Flask, flash, jsonify, redirect, render_template,
    request, session, url_for
)
from tensorflow import keras
from werkzeug.security import check_password_hash, generate_password_hash


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rainfall_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CITIES_CSV = os.path.join(BASE_DIR, "data", "worldcities.csv")
DB_PATH = os.path.join(BASE_DIR, "database", "users.db")

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# True  -> region dropdown shows "13 - Kerala"
# False -> region dropdown shows just "Kerala"
SHOW_REGION_CODE = True

FEATURES = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "cloud_cover"
]

# Smallest alert limit (mm) that the user may enter
MIN_ALERT_LIMIT = 10

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "change-this-secret-key"
)


# ==================================================
# LOAD MODEL AND SCALER
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

model = keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

if hasattr(scaler, "n_features_in_"):
    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"Your scaler expects {scaler.n_features_in_} features, "
            f"but this app uses {len(FEATURES)}. "
            "Use the scaler and model trained together."
        )


# ==================================================
# DATABASE
# ==================================================

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns every prediction_history table must have (used to upgrade old tables)
HISTORY_COLUMNS = {
    "location": "TEXT",
    "temperature": "REAL",
    "humidity": "REAL",
    "pressure": "REAL",
    "wind_speed": "REAL",
    "cloud_cover": "REAL",
    "predicted_rainfall": "REAL",
    "alert_threshold": "REAL",
    "created_at": "TIMESTAMP",
}


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with closing(get_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email TEXT UNIQUE COLLATE NOCASE,
                password TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                location TEXT NOT NULL,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                cloud_cover REAL,
                predicted_rainfall REAL NOT NULL,
                alert_threshold REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Add any columns missing from an older version of the table
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(prediction_history)")
        }
        for column, col_type in HISTORY_COLUMNS.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE prediction_history "
                    f"ADD COLUMN {column} {col_type}"
                )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_user "
            "ON prediction_history(user_id, id)"
        )

        conn.commit()


init_db()


# ==================================================
# COUNTRY NORMALIZATION
# ==================================================

COUNTRY_CODE_TO_NAME = {}
COUNTRY_NAME_TO_CODE = {}

for _country in pycountry.countries:
    _friendly_name = getattr(_country, "common_name", _country.name)
    COUNTRY_CODE_TO_NAME[_country.alpha_2] = _friendly_name
    COUNTRY_NAME_TO_CODE[_country.name.casefold()] = _country.alpha_2
    COUNTRY_NAME_TO_CODE[_friendly_name.casefold()] = _country.alpha_2

    if hasattr(_country, "alpha_3"):
        COUNTRY_NAME_TO_CODE[_country.alpha_3.casefold()] = _country.alpha_2
        COUNTRY_CODE_TO_NAME[_country.alpha_3] = _friendly_name

COUNTRY_NAME_TO_CODE.update({
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "russia": "RU",
    "south korea": "KR",
    "north korea": "KP",
    "iran": "IR",
    "syria": "SY",
    "tanzania": "TZ",
    "bolivia": "BO",
    "venezuela": "VE",
    "moldova": "MD",
    "laos": "LA",
    "brunei": "BN",
    "ivory coast": "CI",
    "czech republic": "CZ",
    "palestine": "PS",
    "taiwan": "TW",
    "kosovo": "XK",
})

COUNTRY_CODE_TO_NAME["XK"] = "Kosovo"


def normalize_country_code(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value or value.casefold() in {"nan", "none", "null"}:
        return None

    upper_value = value.upper()

    if upper_value in COUNTRY_CODE_TO_NAME and len(upper_value) == 2:
        return upper_value

    country = pycountry.countries.get(alpha_3=upper_value)
    if country:
        return country.alpha_2

    return COUNTRY_NAME_TO_CODE.get(value.casefold())


def country_name(code):
    if not code:
        return ""

    code = str(code).strip().upper()

    if code in COUNTRY_CODE_TO_NAME:
        return COUNTRY_CODE_TO_NAME[code]

    country = pycountry.countries.get(alpha_2=code)
    return country.name if country else code


# ==================================================
# CITY / REGION HELPERS
# ==================================================

# Region values such as "01", "03", "CA", "H9" are codes, not names.
CODE_PATTERN = r"[A-Za-z0-9]{1,3}"

REGION_LOCK = threading.Lock()
REGIONS_DONE = set()          # countries whose region names are ready
REGIONS_ARE_CODES = False     # set when the CSV stores region CODES
_WARNED_NO_GEOCODER = False


def make_city_key(name):
    """Key used to merge duplicates like 'Barnes Hill' / 'Barnes Hill Village'."""
    key = str(name).casefold().strip()
    key = re.sub(r"\s+(village|town|city)$", "", key)
    key = re.sub(r"[^a-z0-9]", "", key)
    return key


def clean_place_name(name):
    """'Thrissur District' -> 'Thrissur', 'Kerala State' -> 'Kerala'."""
    name = str(name).strip()
    name = re.sub(r"^(district|state|province)\s+of\s+", "", name, flags=re.I)
    name = re.sub(
        r"\s+(district|dist\.?|state|province|region)$", "", name, flags=re.I
    )
    return name.strip()


def split_region_codes(df):
    """
    If the CSV region column holds codes (01, 03, CA ...), move them to
    'admin_code' and leave 'admin_name' empty until the real name is found.
    If it already holds real names, keep them as they are.
    """
    looks_like_code = df["admin_name"].str.fullmatch(CODE_PATTERN, na=False)
    non_empty = df["admin_name"] != ""

    are_codes = bool(
        non_empty.any() and looks_like_code[non_empty].mean() > 0.8
    )

    if are_codes:
        df["admin_code"] = df["admin_name"].where(looks_like_code, "")
        df["admin_name"] = df["admin_name"].where(~looks_like_code, "")
    else:
        df["admin_code"] = ""

    return df, are_codes


def refresh_region_labels(df, idx=None):
    """Dropdown text such as '03 - Saint John' (or just the code / name)."""
    rows = df.index if idx is None else idx

    code = df.loc[rows, "admin_code"].fillna("")
    name = df.loc[rows, "admin_name"].fillna("")

    if SHOW_REGION_CODE:
        both = code + " - " + name
    else:
        both = name

    df.loc[rows, "admin_label"] = np.where(
        (code != "") & (name != ""), both,
        np.where(code != "", code, name)
    )


def ensure_regions(country):
    """
    Finds the real region/state names for ONE country, the first time it is
    needed, and keeps them in memory. Nothing is written to disk.

    Needs:  pip install reverse_geocoder
    """
    global _WARNED_NO_GEOCODER

    if not REGIONS_ARE_CODES or country in REGIONS_DONE:
        return

    with REGION_LOCK:
        if country in REGIONS_DONE:
            return

        try:
            import reverse_geocoder as rg
        except ImportError:
            if not _WARNED_NO_GEOCODER:
                print(
                    "[warning] 'reverse_geocoder' is not installed, so "
                    "region names cannot be found. Run: "
                    "pip install reverse_geocoder"
                )
                _WARNED_NO_GEOCODER = True
            return

        mask = (
            (cities_df["country"] == country)
            & (cities_df["admin_code"] != "")
        )
        idx = cities_df.index[mask]

        if len(idx) == 0:
            REGIONS_DONE.add(country)
            return

        try:
            coords = list(zip(
                cities_df.loc[idx, "lat"].tolist(),
                cities_df.loc[idx, "lng"].tolist()
            ))

            # mode=1 = single process (safe on Windows).
            results = rg.search(coords, mode=1)
        except Exception as error:
            print(f"[warning] Region name lookup failed for {country}: {error}")
            return

        found = pd.DataFrame({
            "admin_code": cities_df.loc[idx, "admin_code"].to_numpy(),
            "name": [r.get("admin1", "") for r in results],
            "cc": [str(r.get("cc", "")).upper() for r in results],
        })

        country_code = cities_df.loc[idx[0], "country_code"]
        found = found[(found["cc"] == country_code) & (found["name"] != "")]

        # One name per code: the most common name among that code's cities.
        code_to_name = (
            found.groupby("admin_code")["name"]
            .agg(lambda names: names.mode().iat[0])
            .to_dict()
        )

        cities_df.loc[idx, "admin_name"] = (
            cities_df.loc[idx, "admin_code"].map(code_to_name).fillna("")
        )
        refresh_region_labels(cities_df, idx)

        REGIONS_DONE.add(country)
        print(f"[regions] Region names ready for {country}.")


# ==================================================
# LOAD WORLD CITIES CSV  (nothing is saved to disk)
# ==================================================

def load_csv_cities():
    if not os.path.exists(CITIES_CSV):
        raise FileNotFoundError(
            f"Cities CSV not found: {CITIES_CSV}\n"
            "Put worldcities.csv inside your data folder."
        )

    df = pd.read_csv(
        CITIES_CSV,
        encoding="latin1",
        low_memory=False
    )

    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "city": ["city", "city_ascii", "name"],
        "accentcity": ["accentcity", "accent_city"],
        "country_code": ["iso2", "country_code", "country"],
        "lat": ["lat", "latitude"],
        "lng": ["lng", "lon", "longitude"],
        "admin_name": ["admin_name", "state_name", "region", "admin1"],
        "population": ["population", "pop"]
    }

    rename = {}

    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = target
                break

    df = df.rename(columns=rename)

    required = ["city", "country_code", "lat", "lng"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if "accentcity" in df.columns:
        df["city"] = df["accentcity"].fillna(df["city"])

    if "admin_name" not in df.columns:
        df["admin_name"] = ""

    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["admin_name"] = df["admin_name"].fillna("").astype(str).str.strip()

    # Normalize each UNIQUE country value once (much faster than per row).
    raw_codes = df["country_code"].astype(str)
    lookup = {c: normalize_country_code(c) for c in raw_codes.unique()}
    df["country_code"] = raw_codes.map(lookup)

    df = df.dropna(subset=["country_code"]).copy()
    df["country"] = df["country_code"].map(country_name)

    # Coordinates.
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    df = df.dropna(subset=["lat", "lng", "country"])

    df = df[
        (df["city"] != "")
        & df["lat"].between(-90, 90)
        & df["lng"].between(-180, 180)
    ].copy()

    # Biggest cities first, so duplicates keep the most important row.
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        df = df.sort_values(
            "population", ascending=False, na_position="last"
        )
    else:
        df["population"] = np.nan

    df["city_key"] = df["city"].map(make_city_key)
    df = df[df["city_key"] != ""]

    # Region column: codes -> admin_code, real names stay in admin_name.
    df, are_codes = split_region_codes(df)

    # Remove exact duplicates inside the same country and region.
    df = df.drop_duplicates(
        ["country_code", "admin_code", "admin_name", "city_key"]
    )

    df = df.sort_values(
        "population", ascending=False, na_position="last"
    ).reset_index(drop=True)

    df["city_id"] = df.index.astype(int)
    df["admin_label"] = ""

    df = df[[
        "city_id", "city", "city_key", "admin_code", "admin_name",
        "admin_label", "country_code", "country", "lat", "lng",
        "population"
    ]]

    refresh_region_labels(df)

    print(
        f"[startup] Loaded {len(df)} cities from "
        f"{df['country'].nunique()} countries."
    )

    return df, are_codes


# "district" -> third dropdown lists districts, e.g. Kerala -> Thrissur,
#               Palakkad, Kottayam, Alappuzha ... (14 districts)
# "city"     -> third dropdown lists cities and towns instead
LOCATION_LEVEL = "district"

# Only cities with at least this many people are listed (no villages).
# Raise it (for example 50000) to get fewer, bigger cities only.
MIN_CITY_POPULATION = 15000

# Names shown to people instead of the official GeoNames spelling.
CITY_ALIASES = {
    ("IN", "Bengaluru"): "Bangalore",
    ("IN", "Bangalore Urban"): "Bangalore",
}


def load_geonames_cities():
    """
    Clean place list (country -> state/region -> city) taken from the
    GeoNames data that ships inside the 'reverse_geocoder' package.
    Every place already has a real state/region NAME, so nothing has to be
    converted and nothing is written to disk.
    """
    import reverse_geocoder as rg

    path = os.path.join(os.path.dirname(rg.__file__), "rg_cities1000.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoNames file not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    needed = {"lat", "lon", "name", "admin1", "cc"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Unexpected GeoNames columns: {df.columns.tolist()}")

    df = df.rename(columns={
        "lon": "lng", "name": "city",
        "admin1": "admin_name", "cc": "country_code"
    })

    df["country_code"] = df["country_code"].str.strip().str.upper()
    lookup = {c: normalize_country_code(c) for c in df["country_code"].unique()}
    df["country_code"] = df["country_code"].map(lookup)

    df = df.dropna(subset=["country_code"]).copy()
    df["country"] = df["country_code"].map(country_name)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"])

    df["city"] = [
        CITY_ALIASES.get((cc, name.strip()), name.strip())
        for cc, name in zip(df["country_code"], df["city"])
    ]
    df["admin_name"] = df["admin_name"].str.strip()
    df = df[df["city"] != ""].copy()

    df["city_key"] = df["city"].map(make_city_key)
    df = df[df["city_key"] != ""]

    df = df.drop_duplicates(["country_code", "admin_name", "city_key"])

    df = df.sort_values(
        ["country", "admin_name", "city"]
    ).reset_index(drop=True)

    df["city_id"] = df.index.astype(int)
    df["admin_code"] = ""
    df["admin_label"] = df["admin_name"]
    df["population"] = np.nan

    df = df[[
        "city_id", "city", "city_key", "admin_code", "admin_name",
        "admin_label", "country_code", "country", "lat", "lng",
        "population"
    ]]

    print(
        f"[startup] Loaded {len(df)} places from "
        f"{df['country'].nunique()} countries (GeoNames data)."
    )

    return df, False


def load_major_cities():
    """
    Country -> state/region -> city list with real cities only.

    Cities come from the 'geonamescache' package (population based, so no
    villages) and each city's state/region NAME is found with
    'reverse_geocoder'. Nothing is written to disk.

    Needs:  pip install geonamescache reverse_geocoder
    """
    import geonamescache
    import reverse_geocoder as rg

    rows = list(geonamescache.GeonamesCache().get_cities().values())

    df = pd.DataFrame(rows)[[
        "name", "latitude", "longitude", "countrycode", "population"
    ]].rename(columns={
        "name": "city", "latitude": "lat", "longitude": "lng",
        "countrycode": "country_code"
    })

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df[df["population"] >= MIN_CITY_POPULATION]

    df["country_code"] = df["country_code"].astype(str).str.upper()
    lookup = {c: normalize_country_code(c) for c in df["country_code"].unique()}
    df["country_code"] = df["country_code"].map(lookup)

    df = df.dropna(subset=["country_code"]).copy()
    df["country"] = df["country_code"].map(country_name)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    df["city"] = [
        CITY_ALIASES.get((cc, str(name).strip()), str(name).strip())
        for cc, name in zip(df["country_code"], df["city"])
    ]
    df = df[df["city"] != ""].reset_index(drop=True)

    # State / region NAME for every city (offline).
    results = rg.search(
        list(zip(df["lat"].tolist(), df["lng"].tolist())), mode=1
    )

    region = pd.Series([r.get("admin1", "") for r in results], dtype="object")
    same_country = (
        pd.Series([str(r.get("cc", "")).upper() for r in results])
        == df["country_code"]
    )

    df["admin_name"] = region.where(same_country, "").fillna("").str.strip()
    df.loc[df["admin_name"] == "", "admin_name"] = "Other"

    df["city_key"] = df["city"].map(make_city_key)
    df = df[df["city_key"] != ""]

    df = df.drop_duplicates(["country_code", "admin_name", "city_key"])

    df = df.sort_values(
        ["country", "admin_name", "city"]
    ).reset_index(drop=True)

    df["city_id"] = df.index.astype(int)
    df["admin_code"] = ""
    df["admin_label"] = df["admin_name"]

    df = df[[
        "city_id", "city", "city_key", "admin_code", "admin_name",
        "admin_label", "country_code", "country", "lat", "lng",
        "population"
    ]]

    print(
        f"[startup] Loaded {len(df)} cities (population "
        f"{MIN_CITY_POPULATION}+) from {df['country'].nunique()} countries."
    )

    return df, False


def load_districts():
    """
    Country -> state/region -> DISTRICT list, for example
    India -> Kerala -> Thrissur, Palakkad, Kottayam, Alappuzha ...

    Uses the GeoNames data inside the 'reverse_geocoder' package, which
    stores the state (admin1) and district (admin2) of every place.
    Where a country has no districts, the place names are used instead.
    Nothing is written to disk.
    """
    import reverse_geocoder as rg

    path = os.path.join(os.path.dirname(rg.__file__), "rg_cities1000.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoNames file not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    needed = {"lat", "lon", "name", "admin1", "admin2", "cc"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Unexpected GeoNames columns: {df.columns.tolist()}")

    df = df.rename(columns={
        "lon": "lng", "name": "place",
        "admin1": "admin_name", "cc": "country_code"
    })

    df["country_code"] = df["country_code"].str.strip().str.upper()
    lookup = {c: normalize_country_code(c) for c in df["country_code"].unique()}
    df["country_code"] = df["country_code"].map(lookup)

    df = df.dropna(subset=["country_code"]).copy()
    df["country"] = df["country_code"].map(country_name)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"])

    df["place"] = df["place"].map(clean_place_name)
    df["admin_name"] = (
        df["admin_name"].map(clean_place_name).replace("", "Other")
    )
    df["admin2"] = df["admin2"].map(clean_place_name)

    # District name; if the country has none, use the place name.
    df["city"] = df["admin2"].where(df["admin2"] != "", df["place"])
    df["city"] = [
        CITY_ALIASES.get((cc, name), name)
        for cc, name in zip(df["country_code"], df["city"])
    ]
    df = df[df["city"] != ""]

    # A place whose name equals its district is the best map point.
    df["_is_main"] = df["place"].str.casefold() == df["city"].str.casefold()
    df = df.sort_values("_is_main", ascending=False)

    group_cols = ["country_code", "country", "admin_name", "city"]
    main = df.groupby(group_cols, sort=False).first().reset_index()
    middle = (
        df.groupby(group_cols, sort=False)[["lat", "lng"]]
        .median().reset_index()
    )

    # Use the main place when there is one, otherwise the district middle.
    main = main.merge(
        middle, on=group_cols, suffixes=("", "_mid")
    )
    use_main = main["_is_main"]
    main["lat"] = np.where(use_main, main["lat"], main["lat_mid"])
    main["lng"] = np.where(use_main, main["lng"], main["lng_mid"])

    df = main[group_cols + ["lat", "lng"]].copy()

    df["city_key"] = df["city"].map(make_city_key)
    df = df[df["city_key"] != ""]
    df = df.drop_duplicates(["country_code", "admin_name", "city_key"])

    df = df.sort_values(
        ["country", "admin_name", "city"]
    ).reset_index(drop=True)

    df["city_id"] = df.index.astype(int)
    df["admin_code"] = ""
    df["admin_label"] = df["admin_name"]
    df["population"] = np.nan

    df = df[[
        "city_id", "city", "city_key", "admin_code", "admin_name",
        "admin_label", "country_code", "country", "lat", "lng",
        "population"
    ]]

    print(
        f"[startup] Loaded {len(df)} districts from "
        f"{df['country'].nunique()} countries (GeoNames data)."
    )

    return df, False


def load_cities():
    if LOCATION_LEVEL == "district":
        try:
            return load_districts()
        except Exception as error:
            print(
                f"[warning] Could not build the district list ({error}).\n"
                "          Trying the city list instead."
            )

    try:
        return load_major_cities()
    except Exception as error:
        print(
            f"[warning] Could not build the city list ({error}).\n"
            "          Run: pip install geonamescache reverse_geocoder"
        )

    try:
        return load_geonames_cities()
    except Exception as error:
        print(
            f"[warning] Could not use the GeoNames place list ({error}).\n"
            "          Falling back to data/worldcities.csv."
        )

    return load_csv_cities()


cities_df, REGIONS_ARE_CODES = load_cities()
COUNTRIES = sorted(cities_df["country"].dropna().unique())


# ==================================================
# AUTHENTICATION HELPER
# ==================================================

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in first.", "error")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


# ==================================================
# WEATHER AND PREDICTION HELPERS
# ==================================================

def make_place_name(row):
    parts = [
        str(row["city"]).strip(),
        str(row["admin_label"]).strip(),
        str(row["country"]).strip()
    ]

    return ", ".join(
        part for part in parts
        if part and part.casefold() not in {"nan", "none"}
    )


def make_weather_session():
    """Requests session that retries slow or failed weather calls."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    session_ = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session_.mount("https://", adapter)
    session_.mount("http://", adapter)
    return session_


weather_session = make_weather_session()


def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "surface_pressure,wind_speed_10m,cloud_cover"
        )
    }

    # (connect timeout, read timeout) in seconds
    response = weather_session.get(
        WEATHER_URL, params=params, timeout=(15, 30)
    )
    response.raise_for_status()
    data = response.json()

    if "current" not in data:
        raise ValueError("Weather API did not return current data.")

    current = data["current"]

    weather = {
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "pressure": current["surface_pressure"],
        "wind_speed": current["wind_speed_10m"],
        "cloud_cover": current["cloud_cover"]
    }

    if any(
        value is None or not np.isfinite(float(value))
        for value in weather.values()
    ):
        raise ValueError("Weather API returned incomplete data.")

    return weather


def predict_rainfall(weather):
    values = np.array(
        [[weather[feature] for feature in FEATURES]],
        dtype=float
    )

    scaled = scaler.transform(values)
    prediction = model.predict(scaled, verbose=0)

    mm = float(np.asarray(prediction).reshape(-1)[0])

    if not np.isfinite(mm):
        raise ValueError("The model returned an invalid prediction.")

    return round(max(mm, 0.0), 2)


def rainfall_status(mm):
    return "Rain expected" if mm >= 0.1 else "No rain expected"


def rainfall_level(mm):
    if mm < 0.1:
        return "No Rain"
    elif mm < 2.5:
        return "Very Light"
    elif mm < 15.6:
        return "Light"
    elif mm < 64.5:
        return "Moderate"
    elif mm < 115.6:
        return "Heavy"
    elif mm < 204.5:
        return "Very Heavy"

    return "Extremely Heavy"


def build_alert(name, mm, level, threshold):
    if mm >= threshold:
        return (
            True,
            f"Alert for {name}: {level} rainfall of about {mm} mm "
            f"is predicted. Your alert limit is {threshold} mm. "
            "Carry an umbrella and plan ahead."
        )

    return (
        False,
        f"Hi {name}, predicted rainfall is {mm} mm, below your "
        f"alert limit of {threshold} mm."
    )


def save_prediction(user_id, location, weather, mm, threshold):
    with closing(get_db()) as conn:
        conn.execute("""
            INSERT INTO prediction_history (
                user_id, location, temperature, humidity,
                pressure, wind_speed, cloud_cover,
                predicted_rainfall, alert_threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            location,
            weather["temperature"],
            weather["humidity"],
            weather["pressure"],
            weather["wind_speed"],
            weather["cloud_cover"],
            mm,
            threshold
        ))

        conn.commit()


# ==================================================
# LANDING PAGE
# ==================================================

@app.route("/")
def landing():
    return render_template("landing.html")


# ==================================================
# REGISTRATION
# ==================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower() or None
        password = request.form.get("password", "")
        confirm = request.form.get(
            "confirm",
            request.form.get("confirm_password", password)
        )

        if not username or not password:
            flash("Username and password are required.", "error")

        elif len(username) < 3:
            flash("Username must contain at least 3 characters.", "error")

        elif len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")

        elif password != confirm:
            flash("Passwords do not match.", "error")

        else:
            try:
                with closing(get_db()) as conn:
                    conn.execute(
                        "INSERT INTO users (username, email, password) "
                        "VALUES (?, ?, ?)",
                        (username, email, generate_password_hash(password))
                    )
                    conn.commit()

                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))

            except sqlite3.IntegrityError:
                flash("This username or email is already registered.", "error")

    return render_template("register.html")


# ==================================================
# LOGIN
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    if request.method == "POST":
        identifier = (
            request.form.get("username")
            or request.form.get("email")
            or request.form.get("identifier")
            or ""
        ).strip()

        password = request.form.get("password", "")

        with closing(get_db()) as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (identifier, identifier.lower())
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["logged_in"] = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(url_for("home"))

        flash("Invalid username/email or password.", "error")

    return render_template("login.html")


# ==================================================
# LOGOUT AND HOME
# ==================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html", username=session["username"])


@app.route("/prediction")
@login_required
def prediction_page():
    return render_template(
        "predict.html",
        username=session["username"],
        countries=COUNTRIES
    )


# ==================================================
# COUNTRY API
# ==================================================

@app.route("/api/countries")
@login_required
def countries_api():
    return jsonify([
        {"value": country, "label": country}
        for country in COUNTRIES
    ])


# ==================================================
# REGION / STATE API  (shows 'code - name', e.g. 03 - Saint John)
# ==================================================

@app.route("/api/states")
@login_required
def states_api():
    country = request.args.get("country", "").strip()

    ensure_regions(country)

    df = cities_df[cities_df["country"] == country]

    states = sorted({
        str(state).strip()
        for state in df["admin_label"].unique()
        if str(state).strip()
        and str(state).strip().casefold() not in {"nan", "none"}
    })

    return jsonify([
        {"value": state, "label": state}
        for state in states
    ])


# ==================================================
# CITY API  (all cities, duplicates removed, sorted A-Z)
# ==================================================

@app.route("/api/cities")
@login_required
def cities_api():
    country = request.args.get("country", "").strip()
    state = request.args.get("state", "").strip()
    query = request.args.get("q", "").strip().casefold()

    ensure_regions(country)

    df = cities_df[cities_df["country"] == country]

    if state:
        df = df[df["admin_label"] == state]

    df = df.drop_duplicates(["city_key", "admin_label"])

    if query:
        names = df["city"].str.casefold()
        df = df[names.str.contains(query, regex=False)]

        df = df.assign(
            _rank=~df["city"].str.casefold().str.startswith(query)
        ).sort_values("_rank", kind="stable").drop(columns="_rank")
    else:
        df = df.sort_values("city", key=lambda s: s.str.casefold())

    return jsonify([
        {"value": int(row.city_id), "label": str(row.city)}
        for row in df.itertuples()
    ])


# ==================================================
# PREDICT RAINFALL
# ==================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        threshold = float(request.form.get("threshold") or 10)

        if not np.isfinite(threshold) or threshold < MIN_ALERT_LIMIT:
            raise ValueError

    except (ValueError, TypeError):
        flash(
            f"Invalid alert limit. Enter {MIN_ALERT_LIMIT} mm or more.",
            "error"
        )
        return redirect(url_for("prediction_page"))

    try:
        city_id = int(request.form.get("city_id", ""))

        matches = cities_df[cities_df["city_id"] == city_id]

        if matches.empty:
            raise ValueError("Invalid city ID")

        row = matches.iloc[0]
        lat = float(row["lat"])
        lon = float(row["lng"])

    except (ValueError, TypeError, KeyError):
        flash("Please select a valid city from the city dropdown.", "error")
        return redirect(url_for("prediction_page"))

    place_name = make_place_name(row)

    try:
        weather = get_weather(lat, lon)

    except (requests.RequestException, KeyError, ValueError):
        app.logger.exception("Weather API request failed")
        flash(
            "Unable to get live weather data. Check your internet "
            "connection and try again.",
            "error"
        )
        return redirect(url_for("prediction_page"))

    try:
        mm = predict_rainfall(weather)

    except (ValueError, TypeError, KeyError):
        app.logger.exception("Rainfall prediction failed")
        flash(
            "Rainfall prediction failed. Please check your model "
            "and scaler files.",
            "error"
        )
        return redirect(url_for("prediction_page"))

    level = rainfall_level(mm)

    is_alert, alert_text = build_alert(
        session["username"], mm, level, threshold
    )

    try:
        save_prediction(
            session["user_id"], place_name, weather, mm, threshold
        )
    except sqlite3.Error:
        app.logger.exception("Could not save prediction history")
        flash("Prediction done, but it could not be saved to history.", "error")

    return render_template(
        "result.html",
        lat=round(lat, 4),
        lon=round(lon, 4),
        place_name=place_name,
        weather=weather,
        mm=mm,
        status=rainfall_status(mm),
        level=level,
        is_alert=is_alert,
        alert_text=alert_text,
        threshold=threshold
    )


# ==================================================
# PREDICTION HISTORY (SQLite)
# ==================================================

@app.route("/history")
@login_required
def history():
    with closing(get_db()) as conn:
        rows = conn.execute("""
            SELECT id, location, temperature, humidity, pressure,
                   wind_speed, cloud_cover, predicted_rainfall,
                   alert_threshold,
                   datetime(created_at, 'localtime') AS created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 200
        """, (session["user_id"],)).fetchall()

    records = []

    for row in rows:
        record = dict(row)
        mm = record["predicted_rainfall"] or 0
        limit = record["alert_threshold"]

        record["level"] = rainfall_level(mm)
        record["alert"] = limit is not None and mm >= limit

        # How close the prediction is to the user's alert limit (0-100).
        if limit and limit > 0:
            record["pct"] = int(min(100, round(mm / limit * 100)))
        else:
            record["pct"] = 100 if mm > 0 else 0

        records.append(record)

    amounts = [r["predicted_rainfall"] or 0 for r in records]

    stats = {
        "total": len(records),
        "alerts": sum(1 for r in records if r["alert"]),
        "average": round(sum(amounts) / len(amounts), 2) if amounts else 0,
        "highest": round(max(amounts), 2) if amounts else 0,
    }

    return render_template(
        "history.html",
        username=session["username"],
        records=records,
        stats=stats
    )


@app.route("/history/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_history_record(record_id):
    with closing(get_db()) as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE id = ? AND user_id = ?",
            (record_id, session["user_id"])
        )
        conn.commit()
        
    flash("Prediction deleted.", "success")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    with closing(get_db()) as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE user_id = ?",
            (session["user_id"],)
        )
        conn.commit()

    flash("All your saved predictions were deleted.", "success")
    return redirect(url_for("history"))


# ==================================================
# START APPLICATION
# ==================================================

if __name__ == "__main__":
    print("Starting Rainfall Prediction application...")

    try:
        from waitress import serve

        print("Open http://127.0.0.1:5000 in your browser.")
        serve(app, host="127.0.0.1", port=5000)

    except ImportError:
        print("Waitress is not installed; using Flask development server.")
        app.run(debug=True, use_reloader=False)