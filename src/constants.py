"""
constants.py — Project-wide constants shared across pipeline, API, and Streamlit app.
"""

SEASON_MAP: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

SEASON_ORDER: list[str] = ["Winter", "Summer", "Monsoon", "Autumn"]

CITY_NAME_ALIASES: dict[str, str] = {
    "Dacca": "Dhaka",
    "Chattogram": "Chittagong",
    "Chottogram": "Chittagong",
}
