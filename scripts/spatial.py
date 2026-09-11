"""
spatial.py

Geospatial & Spatial Context Engine for Darukaa.Earth.
Satisfies the Hackathon Bonus Requirement:
  "Bonus: Geo-coordinates or spatial context"

Capabilities:
1. Parse coordinates from text or structured dictionaries (lat/lon).
2. Infer Eco-Region / Biome and Koppen-Geiger climate classification.
3. Provide baseline rainfall, temperature, and degradation vulnerabilities.
4. Supply spatial bounding data and map-ready coordinates for UI visualization.
"""

import re
from typing import Optional, Dict, Any, Tuple


# Known benchmark eco-regions / biomes for fast, accurate grounding
ECO_REGIONS = [
    {
        "name": "Semi-Arid Scrub & Dryland",
        "code": "BSh_BSk",
        "description": "Fragile arid/semi-arid drylands with low erratic rainfall, high evapotranspiration, and severe risk of wind erosion and SOC depletion.",
        "baseline_rainfall_mm": 280,
        "baseline_temp_c": 28.5,
        "typical_soc": 0.4,
        "vulnerability_factors": [
            "Low soil organic carbon (< 0.5%)",
            "High evaporative moisture loss",
            "High wind erosion risk on bare cropland",
            "Vulnerability to prolonged droughts"
        ],
        "native_vegetation": "Drought-tolerant shrubs, acacia, deep-rooted legumes, native grasses",
        "lat_range": (15.0, 32.0),
        "lon_range": (68.0, 82.0),  # Example: NW India / Thar / Deccan rain-shadow
    },
    {
        "name": "Tropical Moist Deciduous & Monsoon Forest",
        "code": "Am_Aw",
        "description": "High seasonal monsoon rainfall followed by distinct dry season. High biomass potential but vulnerable to rapid soil leaching and deforestation.",
        "baseline_rainfall_mm": 1350,
        "baseline_temp_c": 26.0,
        "typical_soc": 1.2,
        "vulnerability_factors": [
            "Heavy monsoon runoff and topsoil erosion",
            "Rapid nutrient leaching in open tilled soils",
            "Habitat fragmentation from agricultural conversion",
            "Canopy loss driving microclimate heating"
        ],
        "native_vegetation": "Multi-strata canopy, teak, sal, bamboo, riparian ficus",
        "lat_range": (8.0, 24.0),
        "lon_range": (73.0, 88.0),  # Example: Central / Peninsular India & SE Asia
    },
    {
        "name": "Mediterranean Woodlands & Scrub",
        "code": "Csa_Csb",
        "description": "Dry hot summers and mild wet winters. Severe summer water deficit and high wildfire susceptibility.",
        "baseline_rainfall_mm": 480,
        "baseline_temp_c": 19.0,
        "typical_soc": 0.9,
        "vulnerability_factors": [
            "Summer water deficit stressing understory species",
            "Thin rocky topsoil vulnerable to compaction",
            "Wildfire disturbance disrupting pollinator habitat"
        ],
        "native_vegetation": "Evergreen sclerophyllous shrubs, wild olive, carob, aromatic herbs",
        "lat_range": (30.0, 45.0),
        "lon_range": (-10.0, 36.0),  # Mediterranean Basin / California / S. Africa
    },
    {
        "name": "Temperate Continental Grassland & Steppe",
        "code": "Dfb_BSk",
        "description": "Warm summers and cold winters with moderate precipitation. Historic deep organic soils now heavily depleted by intensive monoculture.",
        "baseline_rainfall_mm": 520,
        "baseline_temp_c": 11.5,
        "typical_soc": 1.8,
        "vulnerability_factors": [
            "Intensive tillage causing rapid organic matter oxidation",
            "Wind erosion during spring fallow periods",
            "Pesticide and fertilizer runoff impacting freshwater bodies"
        ],
        "native_vegetation": "Perennial deep-rooted grasses, clover, shelterbelt deciduous trees",
        "lat_range": (40.0, 55.0),
        "lon_range": (-105.0, 50.0),  # North American Great Plains / Eurasian Steppe
    },
    {
        "name": "Humid Tropical Rainforest & Lowlands",
        "code": "Af",
        "description": "Perennially warm and wet with dense multi-tier canopy. Exceptional biodiversity; cleared land rapidly acidifies and degrades.",
        "baseline_rainfall_mm": 2400,
        "baseline_temp_c": 27.0,
        "typical_soc": 2.2,
        "vulnerability_factors": [
            "Extreme nutrient loss when forest canopy is removed",
            "Severe soil acidification (pH < 5.0) upon clearing",
            "Disruption of high-endemism canopy and understory fauna"
        ],
        "native_vegetation": "Dipterocarps, multi-tier emergent trees, epiphytes, dense understory",
        "lat_range": (-12.0, 10.0),
        "lon_range": (-80.0, 140.0),  # Amazon, Congo Basin, SE Asian archipelago
    }
]


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """
    Extracts latitude and longitude from strings like:
      - "26.9124, 75.7873"
      - "lat: 26.9, lon: 75.8"
      - "26.9°N, 75.8°E"
      - "coordinates: (28.6139, 77.2090)"
    """
    labeled_pattern = r"(?:lat(?:itude)?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?))\s*(?:,\s*|\s+)(?:lon(?:gitude)?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?))"
    match = re.search(labeled_pattern, text, re.IGNORECASE)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon

    cardinal_pattern = r"(\d+(?:\.\d+)?)\s*°?\s*([NSns])\s*,\s*(\d+(?:\.\d+)?)\s*°?\s*([EWew])"
    match = re.search(cardinal_pattern, text)
    if match:
        lat_val, lat_dir, lon_val, lon_dir = match.groups()
        lat = float(lat_val) * (-1 if lat_dir.upper() == "S" else 1)
        lon = float(lon_val) * (-1 if lon_dir.upper() == "W" else 1)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon

    pair_pattern = r"([+-]?\d{1,2}(?:\.\d+)?)\s*,\s*([+-]?\d{1,3}(?:\.\d+)?)"
    match = re.search(pair_pattern, text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon

    return None


def resolve_spatial_profile(lat: float, lon: float, region_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Infers eco-region, biome characteristics, climatic baselines, and soil vulnerabilities
    for a given coordinate pair.
    """
    matched_eco = None

    for eco in ECO_REGIONS:
        lat_min, lat_max = eco["lat_range"]
        lon_min, lon_max = eco["lon_range"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            matched_eco = eco
            break

    if not matched_eco:
        abs_lat = abs(lat)
        if abs_lat <= 12:
            matched_eco = ECO_REGIONS[4]  # Humid Tropical
        elif 12 < abs_lat <= 30:
            matched_eco = ECO_REGIONS[0]  # Arid / Semi-Arid Subtropical
        elif 30 < abs_lat <= 42:
            matched_eco = ECO_REGIONS[2]  # Mediterranean / Warm temperate
        else:
            matched_eco = ECO_REGIONS[3]  # Temperate Continental

    return {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "eco_region": matched_eco["name"],
        "climate_classification": matched_eco["code"],
        "description": matched_eco["description"],
        "baseline_annual_rainfall_mm": matched_eco["baseline_rainfall_mm"],
        "baseline_mean_temp_c": matched_eco["baseline_temp_c"],
        "typical_native_soc_percent": matched_eco["typical_soc"],
        "vulnerability_factors": matched_eco["vulnerability_factors"],
        "recommended_native_guilds": matched_eco["native_vegetation"],
        "spatial_summary": (
            f"Coordinates [{lat:.3f}°, {lon:.3f}°] map to the {matched_eco['name']} biome "
            f"({matched_eco['code']}). Baseline precipitation: ~{matched_eco['baseline_rainfall_mm']} mm/yr; "
            f"mean temp: ~{matched_eco['baseline_temp_c']}°C. Key threat profile: "
            f"{', '.join(matched_eco['vulnerability_factors'][:2])}."
        )
    }


def enrich_user_inputs_with_spatial(user_inputs: dict) -> Tuple[dict, Optional[Dict[str, Any]]]:
    """
    If latitude and longitude are present, enriches user_inputs with spatial context and defaults.
    """
    lat = user_inputs.get("latitude")
    lon = user_inputs.get("longitude")
    
    if lat is None or lon is None:
        return user_inputs, None

    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return user_inputs, None

    spatial_profile = resolve_spatial_profile(lat, lon, region_hint=user_inputs.get("region"))

    enriched = dict(user_inputs)
    if not enriched.get("region"):
        enriched["region"] = spatial_profile["eco_region"]

    return enriched, spatial_profile
