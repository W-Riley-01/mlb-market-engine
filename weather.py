"""
Weather + HR-conditions module for MLB game cards.

Fetches first-pitch forecast from Open-Meteo (free, no API key) and
scores ball-carry conditions. The scoring is calibrated to published
baseball physics — specifically Alan Nathan's sensitivities for a
~400 ft fly ball:

  Temperature : +3.3 ft per 10F increase           (Nathan 2012+)
  Humidity    : +1 ft per 50% RH increase          (Nathan / Hardball Times)
  Elevation   : ~30 ft of extra carry at Coors     (Nathan 2007)
  Pressure    : ~1 ft per 0.1 inHg drop            (Nathan)
  Wind        : +19 ft per 5 mph blowing out (~3.8 ft/mph on CF axis)

Temperature, humidity, elevation, and pressure all work through a
single mechanism — air density — so we compute density directly from
the moist-air ideal gas law and let that one number absorb all four
inputs. Wind is independent of density, so it gets its own term.

Each score point corresponds to ~1 ft of carry, centered at 50 for
MLB-standard conditions (70F, 50% RH, 1013.25 hPa, sea level, calm).
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests


# ==========================================
# PHYSICS CONSTANTS
# ==========================================
# Baseball-neutral reference atmosphere. Score = 50 at these values +
# zero wind. Chosen at 70F / 50% RH / sea level because that's the MLB
# humidor spec and roughly the "typical game" most fans calibrate to.
# (ICAO standard is 15C / 59F / 0% RH, which is colder than a typical
# game and would make every summer game look HR-friendly.)
TEMP_REF_F       = 70.0
RH_REF_PCT       = 50.0
PRESSURE_REF_HPA = 1013.25
RHO_REF_KG_M3    = 1.1944     # precomputed density at reference conditions

# Empirical drag-sensitivity factor. Carry distance does not scale 1:1
# with density because gravity still acts over the flight; ~40% of the
# density change shows up as distance change. This value reproduces
# Nathan's 3.3 ft / 10F and 30 ft Coors-vs-sea-level within rounding.
DRAG_K          = 0.40
REFERENCE_DIST  = 400.0   # ft; a typical HR-like fly ball

# Wind along the CF axis is a separate, dominant term.
# 3.8 ft/mph matches Nathan's "5 mph out = 19 ft".
WIND_FT_PER_MPH = 3.8

# Molar masses (kg/mol) and gas constant (J/(mol*K))
M_DRY_AIR = 0.0289644
M_VAPOR   = 0.018015
R_GAS     = 8.31446


# ==========================================
# STADIUM METADATA
# ==========================================
# cf_bearing = compass bearing (degrees CW from true north) of the
#   home-plate -> center-field axis. Used to project wind velocity onto
#   the "out to CF" component.
# tz = IANA timezone for first-pitch clock display.
STADIUM_INFO = {
    'ARI': {'lat': 33.4455, 'lon': -112.0667, 'elevation_m': 331,  'dome': True,  'cf_bearing': 23,  'tz': 'America/Phoenix',     'park': 'Chase Field'},
    'ATL': {'lat': 33.8908, 'lon': -84.4678,  'elevation_m': 305,  'dome': False, 'cf_bearing': 25,  'tz': 'America/New_York',    'park': 'Truist Park'},
    'BAL': {'lat': 39.2840, 'lon': -76.6215,  'elevation_m': 13,   'dome': False, 'cf_bearing': 60,  'tz': 'America/New_York',    'park': 'Camden Yards'},
    'BOS': {'lat': 42.3466, 'lon': -71.0972,  'elevation_m': 6,    'dome': False, 'cf_bearing': 45,  'tz': 'America/New_York',    'park': 'Fenway Park'},
    'CHC': {'lat': 41.9484, 'lon': -87.6553,  'elevation_m': 182,  'dome': False, 'cf_bearing': 40,  'tz': 'America/Chicago',     'park': 'Wrigley Field'},
    'CWS': {'lat': 41.8299, 'lon': -87.6338,  'elevation_m': 181,  'dome': False, 'cf_bearing': 40,  'tz': 'America/Chicago',     'park': 'Rate Field'},
    'CIN': {'lat': 39.0979, 'lon': -84.5072,  'elevation_m': 148,  'dome': False, 'cf_bearing': 72,  'tz': 'America/New_York',    'park': 'Great American Ballpark'},
    'CLE': {'lat': 41.4962, 'lon': -81.6852,  'elevation_m': 200,  'dome': False, 'cf_bearing': 0,   'tz': 'America/New_York',    'park': 'Progressive Field'},
    'COL': {'lat': 39.7559, 'lon': -104.9942, 'elevation_m': 1581, 'dome': False, 'cf_bearing': 15,  'tz': 'America/Denver',      'park': 'Coors Field'},
    'DET': {'lat': 42.3390, 'lon': -83.0485,  'elevation_m': 183,  'dome': False, 'cf_bearing': 120, 'tz': 'America/Detroit',     'park': 'Comerica Park'},
    'HOU': {'lat': 29.7573, 'lon': -95.3555,  'elevation_m': 14,   'dome': True,  'cf_bearing': 345, 'tz': 'America/Chicago',     'park': 'Minute Maid Park'},
    'KC':  {'lat': 39.0517, 'lon': -94.4803,  'elevation_m': 268,  'dome': False, 'cf_bearing': 45,  'tz': 'America/Chicago',     'park': 'Kauffman Stadium'},
    'LAA': {'lat': 33.8003, 'lon': -117.8827, 'elevation_m': 48,   'dome': False, 'cf_bearing': 30,  'tz': 'America/Los_Angeles', 'park': 'Angel Stadium'},
    'LAD': {'lat': 34.0739, 'lon': -118.2400, 'elevation_m': 142,  'dome': False, 'cf_bearing': 25,  'tz': 'America/Los_Angeles', 'park': 'Dodger Stadium'},
    'MIA': {'lat': 25.7783, 'lon': -80.2195,  'elevation_m': 4,    'dome': True,  'cf_bearing': 40,  'tz': 'America/New_York',    'park': 'loanDepot park'},
    'MIL': {'lat': 43.0282, 'lon': -87.9712,  'elevation_m': 181,  'dome': True,  'cf_bearing': 60,  'tz': 'America/Chicago',     'park': 'American Family Field'},
    'MIN': {'lat': 44.9817, 'lon': -93.2776,  'elevation_m': 255,  'dome': False, 'cf_bearing': 90,  'tz': 'America/Chicago',     'park': 'Target Field'},
    'NYM': {'lat': 40.7571, 'lon': -73.8458,  'elevation_m': 4,    'dome': False, 'cf_bearing': 25,  'tz': 'America/New_York',    'park': 'Citi Field'},
    'NYY': {'lat': 40.8296, 'lon': -73.9262,  'elevation_m': 9,    'dome': False, 'cf_bearing': 75,  'tz': 'America/New_York',    'park': 'Yankee Stadium'},
    'OAK': {'lat': 37.7516, 'lon': -122.2005, 'elevation_m': 6,    'dome': False, 'cf_bearing': 55,  'tz': 'America/Los_Angeles', 'park': 'Oakland Coliseum'},
    'PHI': {'lat': 39.9061, 'lon': -75.1665,  'elevation_m': 9,    'dome': False, 'cf_bearing': 20,  'tz': 'America/New_York',    'park': 'Citizens Bank Park'},
    'PIT': {'lat': 40.4469, 'lon': -80.0057,  'elevation_m': 223,  'dome': False, 'cf_bearing': 300, 'tz': 'America/New_York',    'park': 'PNC Park'},
    'SD':  {'lat': 32.7076, 'lon': -117.1570, 'elevation_m': 10,   'dome': False, 'cf_bearing': 0,   'tz': 'America/Los_Angeles', 'park': 'Petco Park'},
    'SF':  {'lat': 37.7786, 'lon': -122.3893, 'elevation_m': 5,    'dome': False, 'cf_bearing': 90,  'tz': 'America/Los_Angeles', 'park': 'Oracle Park'},
    'SEA': {'lat': 47.5914, 'lon': -122.3325, 'elevation_m': 6,    'dome': True,  'cf_bearing': 0,   'tz': 'America/Los_Angeles', 'park': 'T-Mobile Park'},
    'STL': {'lat': 38.6226, 'lon': -90.1928,  'elevation_m': 136,  'dome': False, 'cf_bearing': 65,  'tz': 'America/Chicago',     'park': 'Busch Stadium'},
    'TB':  {'lat': 27.7682, 'lon': -82.6534,  'elevation_m': 13,   'dome': True,  'cf_bearing': 45,  'tz': 'America/New_York',    'park': 'Tropicana Field'},
    'TEX': {'lat': 32.7373, 'lon': -97.0845,  'elevation_m': 171,  'dome': True,  'cf_bearing': 0,   'tz': 'America/Chicago',     'park': 'Globe Life Field'},
    'TOR': {'lat': 43.6414, 'lon': -79.3894,  'elevation_m': 78,   'dome': True,  'cf_bearing': 0,   'tz': 'America/Toronto',     'park': 'Rogers Centre'},
    'WSH': {'lat': 38.8730, 'lon': -77.0074,  'elevation_m': 5,    'dome': False, 'cf_bearing': 30,  'tz': 'America/New_York',    'park': 'Nationals Park'},
}


TEAM_NAME_TO_CODE = {
    'Arizona Diamondbacks':  'ARI',
    'Atlanta Braves':        'ATL',
    'Baltimore Orioles':     'BAL',
    'Boston Red Sox':        'BOS',
    'Chicago Cubs':          'CHC',
    'Chicago White Sox':     'CWS',
    'Cincinnati Reds':       'CIN',
    'Cleveland Guardians':   'CLE',
    'Colorado Rockies':      'COL',
    'Detroit Tigers':        'DET',
    'Houston Astros':        'HOU',
    'Kansas City Royals':    'KC',
    'Los Angeles Angels':    'LAA',
    'Los Angeles Dodgers':   'LAD',
    'Miami Marlins':         'MIA',
    'Milwaukee Brewers':     'MIL',
    'Minnesota Twins':       'MIN',
    'New York Mets':         'NYM',
    'New York Yankees':      'NYY',
    'Oakland Athletics':     'OAK',
    'Athletics':             'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates':    'PIT',
    'San Diego Padres':      'SD',
    'San Francisco Giants':  'SF',
    'Seattle Mariners':      'SEA',
    'St. Louis Cardinals':   'STL',
    'Tampa Bay Rays':        'TB',
    'Texas Rangers':         'TEX',
    'Toronto Blue Jays':     'TOR',
    'Washington Nationals':  'WSH',
}


FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'


# ==========================================
# PHYSICS HELPERS
# ==========================================
def saturation_vapor_pressure_hpa(temp_c: float) -> float:
    """Magnus-Tetens approximation for saturation water-vapor pressure (hPa)."""
    return 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))


def air_density(temp_f: float, humidity_pct: float, surface_pressure_hpa: float) -> float:
    """
    Moist-air density via the ideal gas law with separate partial pressures
    for dry air and water vapor. Returns kg/m^3.

        rho = (P_d*M_d + P_v*M_v) / (R*T_K)

    Uses surface (station) pressure, not sea-level-adjusted pressure, so
    elevation is already baked in. Callers should pass the `surface_pressure`
    field from Open-Meteo, not `pressure_msl`.
    """
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15

    es_hpa = saturation_vapor_pressure_hpa(temp_c)
    pv_hpa = (humidity_pct / 100.0) * es_hpa
    pd_hpa = max(0.0, surface_pressure_hpa - pv_hpa)

    # Convert hPa -> Pa by multiplying by 100
    rho = (pd_hpa * 100.0 * M_DRY_AIR + pv_hpa * 100.0 * M_VAPOR) / (R_GAS * temp_k)
    return rho


def _deg_to_compass(deg: float) -> str:
    """Bearing in degrees -> 8-way compass label (N, NE, E, ...)."""
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((deg % 360 + 22.5) // 45) % 8
    return dirs[idx]


def _wind_to_cf_component(wind_speed_mph: float, wind_from_deg: float,
                          cf_bearing_deg: float) -> float:
    """
    Scalar projection of wind velocity onto the home-plate -> CF axis.
      Positive = blowing OUT to CF (HR helper).
      Negative = blowing IN from CF (HR killer).

    wind_from_deg is meteorological convention: direction wind COMES FROM.
    The direction it blows TOWARD is wind_from_deg+180. Hence:

        out_component = -wind_speed * cos(wind_from - cf_bearing)
    """
    angle_rad = math.radians(wind_from_deg - cf_bearing_deg)
    return -wind_speed_mph * math.cos(angle_rad)


# ==========================================
# SCORING
# ==========================================
def score_hr_conditions(temp_f: float, humidity_pct: float,
                        surface_pressure_hpa: float, wind_speed_mph: float,
                        wind_from_deg: float, cf_bearing_deg: float,
                        is_dome: bool) -> dict:
    """
    Returns the HR-conditions score (~0-100, 50 neutral), tier label, and
    the per-factor breakdown in feet of carry. Dome games are fixed at 50.

    Density contribution (captures temp + humidity + pressure + elevation
    all at once, since they all feed into air density):
        delta_d_density_ft = 400 * (rho_ref - rho) / rho_ref * 0.4

    Wind contribution (independent of density):
        delta_d_wind_ft = 3.8 * wind_out_component_mph
    """
    if is_dome:
        return {
            'score':              50,
            'label':              'Neutral (Dome)',
            'carry_delta_ft':     0.0,
            'density_delta_ft':   0.0,
            'wind_delta_ft':      0.0,
            'wind_out_mph':       0.0,
            'wind_label':         'Indoors',
            'air_density':        RHO_REF_KG_M3,
        }

    # Air-density delta in ft of carry
    rho = air_density(temp_f, humidity_pct, surface_pressure_hpa)
    density_delta_ft = REFERENCE_DIST * (RHO_REF_KG_M3 - rho) / RHO_REF_KG_M3 * DRAG_K

    # Wind delta in ft of carry (along the CF axis only)
    wind_out = _wind_to_cf_component(wind_speed_mph, wind_from_deg, cf_bearing_deg)
    wind_delta_ft = WIND_FT_PER_MPH * wind_out

    carry_delta_ft = density_delta_ft + wind_delta_ft

    # Each point of score ~= 1 ft of carry. Clip at [0, 100].
    score = max(0, min(100, int(round(50 + carry_delta_ft))))

    # Tier breakpoints — each tier is ~10 ft of carry difference.
    if   score >= 65: label = 'Very HR Friendly'
    elif score >= 55: label = 'HR Friendly'
    elif score >= 45: label = 'Neutral'
    elif score >= 35: label = 'HR Suppressed'
    else:             label = 'Very HR Suppressed'

    if abs(wind_out) < 3:
        wind_label = 'Calm (CF axis)'
    elif wind_out > 0:
        wind_label = f'Out to CF ({wind_out:.0f} mph)'
    else:
        wind_label = f'In from CF ({-wind_out:.0f} mph)'

    return {
        'score':              score,
        'label':              label,
        'carry_delta_ft':     round(carry_delta_ft, 1),
        'density_delta_ft':   round(density_delta_ft, 1),
        'wind_delta_ft':      round(wind_delta_ft, 1),
        'wind_out_mph':       round(wind_out, 1),
        'wind_label':         wind_label,
        'air_density':        round(rho, 4),
    }


# ==========================================
# FORECAST FETCH
# ==========================================
def _parse_game_dt(game_datetime_utc) -> datetime | None:
    """Accept an ISO string ('2026-04-23T23:05:00Z') or datetime. Returns
    a tz-aware UTC datetime, or None if unparseable."""
    if game_datetime_utc is None:
        return None
    if isinstance(game_datetime_utc, datetime):
        if game_datetime_utc.tzinfo is None:
            return game_datetime_utc.replace(tzinfo=timezone.utc)
        return game_datetime_utc.astimezone(timezone.utc)
    try:
        s = str(game_datetime_utc).replace('Z', '+00:00')
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _fetch_forecast(lat: float, lon: float, game_dt_utc: datetime) -> dict | None:
    """Open-Meteo call; returns the hourly row closest to first pitch.
    Pulls all the atmospheric variables we need for the density calc."""
    day = game_dt_utc.date().isoformat()
    params = {
        'latitude':         lat,
        'longitude':        lon,
        'hourly':           'temperature_2m,relative_humidity_2m,surface_pressure,'
                            'wind_speed_10m,wind_direction_10m',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit':  'mph',
        'timezone':         'UTC',
        'start_date':       day,
        'end_date':         day,
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            return None

        target = game_dt_utc.replace(tzinfo=None)
        parsed = [datetime.fromisoformat(t) for t in times]
        best_idx = min(range(len(parsed)),
                       key=lambda i: abs((parsed[i] - target).total_seconds()))

        return {
            'temp_f':        hourly['temperature_2m'][best_idx],
            'humidity_pct':  hourly['relative_humidity_2m'][best_idx],
            'pressure_hpa':  hourly['surface_pressure'][best_idx],
            'wind_speed':    hourly['wind_speed_10m'][best_idx],
            'wind_from_deg': hourly['wind_direction_10m'][best_idx],
        }
    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        print(f'[weather] Forecast fetch failed: {e}')
        return None


# ==========================================
# PUBLIC ENTRY POINT
# ==========================================
def get_game_weather(home_team_name: str, game_datetime_utc) -> dict | None:
    """
    Given a full team name and first-pitch UTC datetime, returns a dict
    the UI can render directly. Fields:

      park, is_dome, first_pitch_local,
      temp_f, humidity_pct, pressure_inhg, wind_speed_mph,
      wind_from_compass, wind_out_mph, wind_label,
      carry_delta_ft, density_delta_ft, wind_delta_ft,
      score, label

    Returns None if the team isn't mapped or the forecast call failed.
    """
    code = TEAM_NAME_TO_CODE.get(home_team_name)
    if not code or code not in STADIUM_INFO:
        return None

    stadium = STADIUM_INFO[code]
    game_dt = _parse_game_dt(game_datetime_utc)

    # First pitch in stadium-local clock time
    if game_dt is not None:
        try:
            local_dt = game_dt.astimezone(ZoneInfo(stadium['tz']))
            first_pitch_local = local_dt.strftime('%-I:%M %p').lstrip('0')
        except Exception:
            first_pitch_local = None
    else:
        first_pitch_local = None

    # Domes: synthesize a neutral entry, no API call
    if stadium['dome']:
        hr = score_hr_conditions(
            temp_f=72.0, humidity_pct=50.0,
            surface_pressure_hpa=PRESSURE_REF_HPA,
            wind_speed_mph=0.0, wind_from_deg=0.0,
            cf_bearing_deg=stadium['cf_bearing'],
            is_dome=True,
        )
        return {
            'park':              stadium['park'],
            'is_dome':           True,
            'first_pitch_local': first_pitch_local,
            'temp_f':            72.0,
            'humidity_pct':      50,
            'pressure_inhg':     29.92,
            'wind_speed_mph':    0.0,
            'wind_from_compass': '—',
            'wind_out_mph':      0.0,
            'wind_label':        'Indoors',
            'carry_delta_ft':    0.0,
            'density_delta_ft':  0.0,
            'wind_delta_ft':     0.0,
            'score':             hr['score'],
            'label':             hr['label'],
        }

    if game_dt is None:
        return None

    fc = _fetch_forecast(stadium['lat'], stadium['lon'], game_dt)
    if fc is None:
        return None

    hr = score_hr_conditions(
        temp_f=fc['temp_f'],
        humidity_pct=fc['humidity_pct'],
        surface_pressure_hpa=fc['pressure_hpa'],
        wind_speed_mph=fc['wind_speed'],
        wind_from_deg=fc['wind_from_deg'],
        cf_bearing_deg=stadium['cf_bearing'],
        is_dome=False,
    )

    return {
        'park':              stadium['park'],
        'is_dome':           False,
        'first_pitch_local': first_pitch_local,
        'temp_f':            round(fc['temp_f'], 1),
        'humidity_pct':      int(round(fc['humidity_pct'])),
        'pressure_inhg':     round(fc['pressure_hpa'] / 33.8639, 2),
        'wind_speed_mph':    round(fc['wind_speed'], 1),
        'wind_from_compass': _deg_to_compass(fc['wind_from_deg']),
        'wind_out_mph':      hr['wind_out_mph'],
        'wind_label':        hr['wind_label'],
        'carry_delta_ft':    hr['carry_delta_ft'],
        'density_delta_ft':  hr['density_delta_ft'],
        'wind_delta_ft':     hr['wind_delta_ft'],
        'score':             hr['score'],
        'label':             hr['label'],
    }


# ==========================================
# PROP ADJUSTMENT
# Applies the weather signal to Monte Carlo-derived player props. Only
# props sensitive to fly-ball distance are touched — singles, walks,
# and SBs are independent of carry.
# ==========================================

# HR-rate sensitivity to carry distance. Based on Nathan's Coors prediction
# (~28 ft of carry -> ~27.5% HR rate increase), which gives a clean 1:1
# mapping of ft to percent near the HR threshold.
HR_RATE_PER_FT = 0.010     # 1% HR-rate change per ft of carry delta

# Extra-base hits (2+ TB) are partially HR-sensitive and partially
# "ball hit the wall harder" sensitive. Empirically this effect is
# ~1/3 as strong as the HR effect.
TB_RATE_PER_FT = 0.0033    # ~0.33% TB-rate change per ft

# Multiplier caps prevent extreme conditions from blowing up the output.
# At the caps (+/-30 ft of carry), we allow HR rates to swing +/-30%.
MAX_CARRY_FT = 30.0


def apply_weather_to_props(player_props: dict,
                            carry_delta_ft: float) -> dict:
    """
    Returns a deep copy of player_props with HR and TB probabilities
    adjusted for today's carry conditions. Other probs are passed through
    unchanged.

    The adjustment is multiplicative on the probability, which is a
    reasonable approximation when per-PA rates are small (always true
    for HR, mostly true for 2+ TB). Hits and SBs are NOT adjusted — the
    carry effect is specific to fly-ball distance.

    carry_delta_ft should come from get_game_weather()['carry_delta_ft'].
    Pass 0 (or skip calling this) for dome games.
    """
    if not player_props or carry_delta_ft is None:
        return player_props

    # Clamp so extreme forecasts don't produce absurd outputs. A real
    # Coors day might be +28 ft; beyond that we're extrapolating.
    carry = max(-MAX_CARRY_FT, min(MAX_CARRY_FT, carry_delta_ft))

    hr_mult = 1.0 + carry * HR_RATE_PER_FT
    tb_mult = 1.0 + carry * TB_RATE_PER_FT

    adjusted = {}
    for name, props in player_props.items():
        new_props = dict(props)  # shallow copy of the inner dict
        if '1+ HR' in new_props:
            new_props['1+ HR'] = max(0.0, min(1.0, new_props['1+ HR'] * hr_mult))
        if '2+ TB' in new_props:
            new_props['2+ TB'] = max(0.0, min(1.0, new_props['2+ TB'] * tb_mult))
        adjusted[name] = new_props

    return adjusted


# ==========================================
# SMOKE TEST
# ==========================================
if __name__ == '__main__':
    from datetime import timedelta
    test_dt = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    for team in ['New York Yankees', 'Colorado Rockies', 'Tampa Bay Rays']:
        print(f'\n--- {team} ---')
        result = get_game_weather(team, test_dt)
        if result:
            for k, v in result.items():
                print(f'  {k:22s} {v}')
        else:
            print('  (no data)')