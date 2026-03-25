import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

from datetime import date
from dateutil.relativedelta import relativedelta
from time import sleep

# -----------------------------
# Client setup
# -----------------------------
def get_openmeteo_client(cache_dir="data/api.cache", retries=5, backoff_factor=0.2):
    cache_session = requests_cache.CachedSession(
        cache_dir, expire_after=-1
    )
    retry_session = retry(
        cache_session,
        retries=retries,
        backoff_factor=backoff_factor
    )
    return openmeteo_requests.Client(session=retry_session)


# -----------------------------
# Main data fetch function
# -----------------------------
def fetch_hourly_weather_data(
    start_date: str,
    end_date: str,
    hourly_vars: list,
    latitude: float = -34.9287,   # Adelaide default
    longitude: float = 138.5986,
    timezone: str = "auto",
    save_path: str | None = None
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo Archive API.

    Returns
    -------
    pd.DataFrame with datetime index (local time).
    """

    client = get_openmeteo_client()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": timezone,
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()

    # Build datetime index 
    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"datetime": time_index}

    for i, var in enumerate(hourly_vars):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("datetime")

    if save_path:
        df.to_parquet(save_path)

    return df


def fetch_hourly_weather_range(
    start_year: int,
    end_year: int,
    hourly_vars: list,
    **kwargs
) -> pd.DataFrame:

    dfs = []

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    current = start

    while current <= end:
        period_start = current
        period_end = min(current + relativedelta(months=6) - relativedelta(days=1), end)

        try:
            print(f"Fetching {period_start} → {period_end} ...")

            df = fetch_hourly_weather_data(
                start_date=period_start.isoformat(),
                end_date=period_end.isoformat(),
                hourly_vars=hourly_vars,
                **kwargs
            )

            if df.empty:
                print(f"Empty data: {period_start} → {period_end}")
            else:
                print(f"Success: {period_start} → {period_end} ({len(df)} rows)")
                dfs.append(df)

        except Exception as e:
            print(
                f"Failed: {period_start} → {period_end} | "
                f"{type(e).__name__}: {e}"
            )

        current = period_end + relativedelta(days=1)
        
        sleep(2) # To not getting rate limit caused errors.

    if not dfs:
        raise RuntimeError("No data could be fetched for the given period.")

    return pd.concat(dfs).sort_index()


