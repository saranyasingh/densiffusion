import numpy as np

from densiffusion.aqs import (
    choose_neighbours,
    dense_then_sparse,
    gap_coverage,
    read_daily_csv,
    year_regimes,
)

HEADER = (
    "State Code,County Code,Site Num,POC,Latitude,Longitude,Date Local,"
    "Arithmetic Mean,City Name,State Name,Pollutant Standard\n"
)


def test_daily_csv_counts_each_monitor_day_once_and_averages_monitors():
    rows = [
        "01,003,0010,1,30.5,-87.9,2005-01-01,9.0,Fairhope,Alabama,24-hour 2006",
        "01,003,0010,1,30.5,-87.9,2005-01-01,9.0,Fairhope,Alabama,24-hour 2012",
        "01,003,0010,2,30.5,-87.9,2005-01-01,11.0,Fairhope,Alabama,24-hour 2012",
    ]
    means, sites = read_daily_csv((HEADER + "\n".join(rows)).splitlines(True))
    assert means == {("01-003-0010", "2005-01-01"): 10.0}
    assert sites["01-003-0010"] == (30.5, -87.9, "Fairhope, Alabama")


def calendar(years):
    return np.arange(
        f"{years[0]}-01-01", f"{years[-1] + 1}-01-01", dtype="datetime64[D]"
    )


def test_year_regimes_recognise_daily_and_one_in_three_years():
    dates = calendar([2001, 2002, 2003])
    values = np.full((len(dates), 1), np.nan)
    first = dates < np.datetime64("2002-01-01")
    values[first, 0] = 5.0
    values[~first, 0] = np.where(np.arange((~first).sum()) % 3 == 0, 5.0, np.nan)
    years, codes = year_regimes(values, dates)
    assert list(years) == [2001, 2002, 2003] and codes[0] == "D33"


def test_dense_then_sparse_finds_the_daily_run_before_the_sparse_run():
    years = np.arange(2000, 2012)
    eras = dense_then_sparse("33DDDDD?3336", years)
    assert eras.dense == (2002, 2006) and eras.sparse == (2008, 2011)
    assert dense_then_sparse("3333DDDDDDDD", years) is None


def test_neighbours_need_coverage_in_both_eras_and_are_nearest_first():
    dates = calendar([2001, 2002])
    values = np.full((len(dates), 4), 1.0)
    dense = dates < np.datetime64("2002-01-01")
    values[~dense, 0] = np.where(np.arange((~dense).sum()) % 3 == 0, 1.0, np.nan)
    values[~dense, 2] = np.nan  # silent in the target's sparse era
    data = {
        "values": values,
        "dates": dates,
        "latitude": np.array([40.0, 40.3, 40.1, 40.2]),
        "longitude": np.array([-80.0, -80.0, -80.0, -80.0]),
    }
    eras = dense_then_sparse("D3", np.array([2001, 2002]), min_dense=1)
    neighbours = choose_neighbours(data, 0, eras, max_km=100)
    assert neighbours == [3, 1]
    assert gap_coverage(data, 0, neighbours, eras) == 1.0
