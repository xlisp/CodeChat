import requests
def get_weather(location, unit="celsius", **_):
    r = requests.get(f"https://wttr.in/{location}?format=j1").json()
    c = r["current_condition"][0]
    return {"location": location,
            "temperature": c["temp_C" if unit=="celsius" else "temp_F"],
            "unit": unit, "conditions": c["weatherDesc"][0]["value"]}
EXECUTORS = {"get_weather": get_weather}

