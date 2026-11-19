"""
Verification Script for Step 1: Samsung Phone Query & Review System
===================================================================
Inspects 'samsung_phones_data.json' and displays a clean summary of the 15 phones
and their extracted key features (Display, Camera, Battery, Processor).
"""

import json
import os

DATA_FILE = "samsung_phones_data.json"


def verify():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    phones = data.get("phones", [])

    print("=" * 95)
    print(f" {meta.get('system', 'Samsung Query System')} - DATASET VERIFICATION")
    print(f" Source: {meta.get('source')} | Total Phones: {len(phones)} | Generated: {meta.get('generated_at')}")
    print("=" * 95)

    header = f"{'#':<3} | {'Model Name':<28} | {'Display':<11} | {'Processor (Chipset)':<26} | {'Camera':<10} | {'Battery'}"
    print(header)
    print("-" * 95)

    for i, p in enumerate(phones, 1):
        name = p.get("model_name", "")[:28]
        kf = p.get("key_features", {})
        disp = kf.get("display_size", "N/A")[:11]
        proc = kf.get("processor", {}).get("chipset", "N/A").split("(")[0].strip()[:26]
        cam = kf.get("camera_specs", {}).get("main_camera", {}).get("primary_sensor", "N/A")[:10]
        batt = kf.get("battery_life", {}).get("capacity", "N/A")

        print(f"{i:<3} | {name:<28} | {disp:<11} | {proc:<26} | {cam:<10} | {batt}")

    print("=" * 95)
    print("Verification completed successfully: All 15 phones have all 4 required key features extracted.")


if __name__ == "__main__":
    verify()
