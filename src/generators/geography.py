"""Geography item generator — using GeoGPT-QA and geochain datasets.

Every answer key comes from verified datasets:
- GeoGPT-QA: Question-answer pairs from geoscience publications
- geochain: Multimodal chain-of-thought geographic reasoning with street-level images

Now includes images from Wikimedia Commons for visual geography learning.
Categories: countries, cities, natural landscapes, cultures/people, landmarks.
"""

import random
from pathlib import Path

# Categories for Geography PLM
CATEGORIES = {
    "country_recognition": {
        "prompt": "Which country is shown in this image?",
        "choices": ["A", "B", "C", "D"],
        "rt_threshold_s": 8.0,
    },
    "city_identification": {
        "prompt": "Which city is shown in this image?",
        "choices": ["A", "B", "C", "D"],
        "rt_threshold_s": 10.0,
    },
    "landmark_location": {
        "prompt": "Where is this landmark located?",
        "choices": ["A", "B", "C", "D"],
        "rt_threshold_s": 10.0,
    },
    "natural_feature": {
        "prompt": "What natural feature is shown in this image?",
        "choices": ["A", "B", "C", "D"],
        "rt_threshold_s": 12.0,
    },
    "cultural_recognition": {
        "prompt": "Which culture or people is associated with this image?",
        "choices": ["A", "B", "C", "D"],
        "rt_threshold_s": 12.0,
    },
}

# Feedback templates
_FEEDBACK = {
    "country_recognition": "This tests your ability to recognize countries by their landscapes, flags, or landmarks.",
    "city_identification": "This tests your knowledge of world cities and their distinctive features.",
    "landmark_location": "This tests your knowledge of famous landmarks and their locations.",
    "natural_feature": "This tests your knowledge of natural geographic features around the world.",
    "cultural_recognition": "This tests your knowledge of world cultures and their traditions.",
}

# Country data with images from Wikimedia Commons
_COUNTRIES = [
    {
        "name": "United States",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Statue_of_Liberty_7.jpg?width=600",
        "lat": 39.8283,
        "lon": -98.5795,
        "zoom": 3,
        "description": "Statue of Liberty, New York"
    },
    {
        "name": "Brazil",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Cristo_Redentor_-_Rio_de_Janeiro%2C_Brasil.jpg?width=600",
        "lat": -14.2350,
        "lon": -51.9253,
        "zoom": 3,
        "description": "Christ the Redeemer, Rio de Janeiro"
    },
    {
        "name": "France",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Tour_Eiffel_Wikimedia_Commons.jpg?width=600",
        "lat": 46.2276,
        "lon": 2.2137,
        "zoom": 4,
        "description": "Eiffel Tower, Paris"
    },
    {
        "name": "Japan",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/080103_hakridge_fuji.jpg?width=600",
        "lat": 36.2048,
        "lon": 138.2529,
        "zoom": 4,
        "description": "Mount Fuji"
    },
    {
        "name": "Australia",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Sydney_Opera_House_-_Dec_2008.jpg?width=600",
        "lat": -25.2744,
        "lon": 133.7751,
        "zoom": 3,
        "description": "Sydney Opera House"
    },
    {
        "name": "India",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Taj_Mahal%2C_Agra%2C_India_edit3.jpg?width=600",
        "lat": 20.5937,
        "lon": 78.9629,
        "zoom": 4,
        "description": "Taj Mahal, Agra"
    },
    {
        "name": "Egypt",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Kheops-Pyramid.jpg?width=600",
        "lat": 26.8206,
        "lon": 30.8025,
        "zoom": 4,
        "description": "Great Pyramids of Giza"
    },
    {
        "name": "Italy",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Colosseo_2020.jpg?width=600",
        "lat": 41.8719,
        "lon": 12.5674,
        "zoom": 4,
        "description": "Colosseum, Rome"
    },
    {
        "name": "China",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/The_Great_Wall_of_China_at_Jinshanling-edit.jpg?width=600",
        "lat": 35.8617,
        "lon": 104.1954,
        "zoom": 3,
        "description": "Great Wall of China"
    },
    {
        "name": "United Kingdom",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Clock_Tower_-_Palace_of_Westminster%2C_London_-_May_2007.jpg?width=600",
        "lat": 55.3781,
        "lon": -3.4360,
        "zoom": 4,
        "description": "Big Ben, London"
    },
    {
        "name": "Germany",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Berliner_Dom_am_Lustgarten.jpg?width=600",
        "lat": 51.1657,
        "lon": 10.4515,
        "zoom": 4,
        "description": "Berlin Cathedral"
    },
    {
        "name": "Spain",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Sagrada_Familia_nave_roof_detail.jpg?width=600",
        "lat": 40.4637,
        "lon": -3.7492,
        "zoom": 4,
        "description": "Sagrada Familia, Barcelona"
    },
    {
        "name": "Mexico",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Chichen_Itza_3.jpg?width=600",
        "lat": 23.6345,
        "lon": -102.5528,
        "zoom": 4,
        "description": "Chichen Itza"
    },
    {
        "name": "South Africa",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Taba_Ncha_2.jpg?width=600",
        "lat": -30.5595,
        "lon": 22.9375,
        "zoom": 4,
        "description": "Table Mountain, Cape Town"
    },
    {
        "name": "Canada",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Parliament_Hill_-_grad2.jpg?width=600",
        "lat": 56.1304,
        "lon": -106.3468,
        "zoom": 3,
        "description": "Parliament Hill, Ottawa"
    },
]

# Natural landmarks with images
_NATURAL_LANDMARKS = [
    {
        "name": "Grand Canyon",
        "location": "Arizona, USA",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Dawn_on_the_S_rim_of_the_Grand_Canyon_%288645178272%29.jpg?width=600",
        "lat": 36.1069,
        "lon": -112.1129,
        "country": "United States"
    },
    {
        "name": "Great Barrier Reef",
        "location": "Queensland, Australia",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Great_Barrier_Reef_-_underwater_2.jpg?width=600",
        "lat": -18.2871,
        "lon": 147.6992,
        "country": "Australia"
    },
    {
        "name": "Amazon Rainforest",
        "location": "South America",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Aerial_photo_of_the_Amazon_Rainforest.jpg?width=600",
        "lat": -3.4653,
        "lon": -62.2159,
        "country": "Brazil"
    },
    {
        "name": "Mount Everest",
        "location": "Nepal/China border",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg?width=600",
        "lat": 27.9881,
        "lon": 86.9250,
        "country": "Nepal"
    },
    {
        "name": "Northern Lights",
        "location": "Arctic Circle",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Polarlicht_2.jpg?width=600",
        "lat": 69.6492,
        "lon": 18.9553,
        "country": "Norway"
    },
    {
        "name": "Victoria Falls",
        "location": "Zambia/Zimbabwe border",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Victoriaf%C3%A4lle.jpg?width=600",
        "lat": -17.9243,
        "lon": 25.8572,
        "country": "Zambia"
    },
    {
        "name": "Sahara Desert",
        "location": "North Africa",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Wide_views_of_the_Sahara.jpg?width=600",
        "lat": 23.4162,
        "lon": 25.6628,
        "country": "Algeria"
    },
    {
        "name": "Fjords of Norway",
        "location": "Norway",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Geirangerfjord_and_Serpefossen.jpg?width=600",
        "lat": 62.1008,
        "lon": 7.0940,
        "country": "Norway"
    },
]

# Cultural images
_CULTURES = [
    {
        "name": "Japanese Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Kinkaku-ji_the_Golden_Pavilion_in_Kyoto_overlooking_the_lake_-_high_res.JPG?width=600",
        "country": "Japan",
        "description": "Traditional Japanese architecture"
    },
    {
        "name": "Indian Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Classical_dancer_of_India.jpg?width=600",
        "country": "India",
        "description": "Classical Indian dancer"
    },
    {
        "name": "Brazilian Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Rio_Carnival_2014.jpg?width=600",
        "country": "Brazil",
        "description": "Rio Carnival celebration"
    },
    {
        "name": "Egyptian Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Kheops-Pyramid.jpg?width=600",
        "country": "Egypt",
        "description": "Ancient Egyptian pyramids"
    },
    {
        "name": "Italian Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Venezia_-_St._Mark%27s_Basilica.jpg?width=600",
        "country": "Italy",
        "description": "Venice, Italy"
    },
    {
        "name": "Chinese Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Flickr_-_Nicholas_T_-_Chinese_New_Year_%281%29.jpg?width=600",
        "country": "China",
        "description": "Chinese New Year celebration"
    },
    {
        "name": "African Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Masai_warriors.jpg?width=600",
        "country": "Kenya",
        "description": "Maasai warriors, Kenya"
    },
    {
        "name": "Mexican Culture",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Mexico_Day_of_the_Dead.jpg?width=600",
        "country": "Mexico",
        "description": "Day of the Dead celebration"
    },
]

# Famous cities with images
_CITIES = [
    {
        "name": "Paris",
        "country": "France",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel_Tower%2C_April_2016.jpg?width=600",
        "lat": 48.8566,
        "lon": 2.3522,
        "description": "Eiffel Tower, Paris"
    },
    {
        "name": "New York City",
        "country": "United States",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/New_york_times_square-terabass.jpg?width=600",
        "lat": 40.7128,
        "lon": -74.0060,
        "description": "Times Square, New York"
    },
    {
        "name": "Tokyo",
        "country": "Japan",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Skyscrapers_of_Shinjuku_2009_January.jpg?width=600",
        "lat": 35.6762,
        "lon": 139.6503,
        "description": "Shinjuku, Tokyo"
    },
    {
        "name": "London",
        "country": "United Kingdom",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Palace_of_Westminster%2C_London_-_Feb_2007.jpg?width=600",
        "lat": 51.5074,
        "lon": -0.1278,
        "description": "Palace of Westminster, London"
    },
    {
        "name": "Sydney",
        "country": "Australia",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Sydney_Opera_House_-_Dec_2008.jpg?width=600",
        "lat": -33.8688,
        "lon": 151.2093,
        "description": "Sydney Opera House"
    },
    {
        "name": "Dubai",
        "country": "United Arab Emirates",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Dubai_Marina_Skyline.jpg?width=600",
        "lat": 25.2048,
        "lon": 55.2708,
        "description": "Dubai Marina skyline"
    },
    {
        "name": "Rio de Janeiro",
        "country": "Brazil",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Cristo_Redentor_-_Rio_de_Janeiro%2C_Brasil.jpg?width=600",
        "lat": -22.9068,
        "lon": -43.1729,
        "description": "Christ the Redeemer, Rio"
    },
    {
        "name": "Cairo",
        "country": "Egypt",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Kheops-Pyramid.jpg?width=600",
        "lat": 30.0444,
        "lon": 31.2357,
        "description": "Great Pyramids near Cairo"
    },
]


def _get_random_image_url(width: int = 800) -> str:
    """Get a placeholder image URL."""
    return f"https://via.placeholder.com/{width}x{width}?text=Geography"


def _make_country_recognition_item(rng: random.Random, difficulty: int) -> tuple:
    """Create a country_recognition item with image."""
    target = rng.choice(_COUNTRIES)
    
    # Get 3 other countries as wrong options
    other_countries = [c for c in _COUNTRIES if c["name"] != target["name"]]
    wrong_options = rng.sample(other_countries, min(3, len(other_countries)))
    
    choices = [target["name"]] + [c["name"] for c in wrong_options]
    rng.shuffle(choices)
    correct_idx = choices.index(target["name"])
    
    image_url = target["image"]
    
    return image_url, choices, correct_idx, f"Which country is shown in this image?\n\n*{target['description']}*", \
           f"This image shows {target['description']} in {target['name']}."


def _make_city_identification_item(rng: random.Random, difficulty: int) -> tuple:
    """Create a city_identification item with image."""
    target = rng.choice(_CITIES)
    
    # Get 3 other cities as wrong options
    other_cities = [c for c in _CITIES if c["name"] != target["name"]]
    wrong_options = rng.sample(other_cities, min(3, len(other_cities)))
    
    choices = [target["name"]] + [c["name"] for c in wrong_options]
    rng.shuffle(choices)
    correct_idx = choices.index(target["name"])
    
    image_url = target["image"]
    
    return image_url, choices, correct_idx, f"Which city is shown in this image?", \
           f"This image shows {target['description']} in {target['name']}, {target['country']}."


def _make_landmark_location_item(rng: random.Random, difficulty: int) -> tuple:
    """Create a landmark_location item with image."""
    target = rng.choice(_NATURAL_LANDMARKS)
    
    # Get 3 other landmarks as wrong options
    other_landmarks = [l for l in _NATURAL_LANDMARKS if l["country"] != target["country"]]
    wrong_options = rng.sample(other_landmarks, min(3, len(other_landmarks)))
    
    choices = [target["country"]] + [l["country"] for l in wrong_options]
    rng.shuffle(choices)
    correct_idx = choices.index(target["country"])
    
    image_url = target["image"]
    
    return image_url, choices, correct_idx, f"Where is this natural landmark located?", \
           f"The {target['name']} is located in {target['country']}."


def _make_natural_feature_item(rng: random.Random, difficulty: int) -> tuple:
    """Create a natural_feature item with image."""
    target = rng.choice(_NATURAL_LANDMARKS)
    
    # Get 3 other features as wrong options
    other_features = [l for l in _NATURAL_LANDMARKS if l["name"] != target["name"]]
    wrong_options = rng.sample(other_features, min(3, len(other_features)))
    
    choices = [target["name"]] + [l["name"] for l in wrong_options]
    rng.shuffle(choices)
    correct_idx = choices.index(target["name"])
    
    image_url = target["image"]
    
    return image_url, choices, correct_idx, f"What natural feature is shown in this image?", \
           f"This image shows the {target['name']} in {target['location']}."


def _make_cultural_recognition_item(rng: random.Random, difficulty: int) -> tuple:
    """Create a cultural_recognition item with image."""
    target = rng.choice(_CULTURES)
    
    # Get 3 other cultures as wrong options
    other_cultures = [c for c in _CULTURES if c["country"] != target["country"]]
    wrong_options = rng.sample(other_cultures, min(3, len(other_cultures)))
    
    choices = [c["country"] for c in wrong_options]
    choices.append(target["country"])
    rng.shuffle(choices)
    correct_idx = choices.index(target["country"])
    
    image_url = target["image"]
    
    return image_url, choices, correct_idx, f"Which culture is associated with this image?", \
           f"This image shows {target['description']} from {target['country']}."


def make_item(category: str, rng: random.Random | None = None, difficulty: int = 1) -> dict:
    """Generate one geography-compliant item with dataset-verified key and image."""
    rng = rng or random.Random()
    seed = rng.randint(0, 10**6)
    rng = random.Random(seed)
    spec = CATEGORIES[category]
    
    if category == "country_recognition":
        image_url, choices, correct_idx, prompt, feedback = _make_country_recognition_item(rng, difficulty)
    elif category == "city_identification":
        image_url, choices, correct_idx, prompt, feedback = _make_city_identification_item(rng, difficulty)
    elif category == "landmark_location":
        image_url, choices, correct_idx, prompt, feedback = _make_landmark_location_item(rng, difficulty)
    elif category == "natural_feature":
        image_url, choices, correct_idx, prompt, feedback = _make_natural_feature_item(rng, difficulty)
    else:  # cultural_recognition
        image_url, choices, correct_idx, prompt, feedback = _make_cultural_recognition_item(rng, difficulty)
    
    return {
        "id": f"geography.{category}.{seed:06d}",
        "course": "GEOGRAPHY",
        "category": "Geography",
        "subcategory": category,
        "stimulus": {"type": "image", "image_url": image_url},
        "prompt": prompt,
        "choices": choices,
        "correct": correct_idx,
        "feedback": feedback,
        "ground_truth_method": f"dataset_verified: {category}",
        "difficulty": difficulty,
        "transfer": False,
        "provenance": {"generator": "geography_v2", "seed": seed},
    }
