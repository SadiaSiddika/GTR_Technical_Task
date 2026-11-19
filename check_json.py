import json

with open("samsung_phones_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total Phones Scraped: {len(data['phones'])}")
for i, phone in enumerate(data["phones"], 1):
    kf = phone["key_features"]
    print(f"{i:2d}. {phone['model_name']:<28} | Display: {kf['display_size']:<10} | Battery: {kf['battery_life']['capacity']}")
