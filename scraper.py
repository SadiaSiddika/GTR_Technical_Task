"""
Samsung Phone Specifications Web Scraper
========================================
Part of Step 1 for the Samsung Phone Query and Review System.
Scrapes comprehensive specifications for Samsung Galaxy phones (S21, S22, S23, S24, etc.)
from GSMArena using requests and BeautifulSoup.

Extracts key features:
- Display size & resolution
- Camera specs (Main & Selfie configurations, lenses, video capabilities)
- Battery life (Capacity, charging, active use & endurance test scores)
- Processor (Chipset, CPU, GPU, OS)
- Full detailed specifications
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SamsungScraper")

# List of 15 target Samsung Galaxy phones spanning S21, S22, S23, S24, FE, and popular A-series
DEFAULT_TARGET_PHONES = [
    {
        "model_name": "Samsung Galaxy S24 Ultra",
        "series": "Galaxy S24 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s24_ultra-12771.php"
    },
    {
        "model_name": "Samsung Galaxy S24+",
        "series": "Galaxy S24 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s24+-12772.php"
    },
    {
        "model_name": "Samsung Galaxy S24",
        "series": "Galaxy S24 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s24-12773.php"
    },
    {
        "model_name": "Samsung Galaxy S23 Ultra",
        "series": "Galaxy S23 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s23_ultra-12024.php"
    },
    {
        "model_name": "Samsung Galaxy S23+",
        "series": "Galaxy S23 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s23+-12083.php"
    },
    {
        "model_name": "Samsung Galaxy S23",
        "series": "Galaxy S23 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s23-12082.php"
    },
    {
        "model_name": "Samsung Galaxy S23 FE",
        "series": "Galaxy S23 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s23_fe-12520.php"
    },
    {
        "model_name": "Samsung Galaxy S22 Ultra 5G",
        "series": "Galaxy S22 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s22_ultra_5g-11251.php"
    },
    {
        "model_name": "Samsung Galaxy S22+ 5G",
        "series": "Galaxy S22 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s22+_5g-11252.php"
    },
    {
        "model_name": "Samsung Galaxy S22 5G",
        "series": "Galaxy S22 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s22_5g-11253.php"
    },
    {
        "model_name": "Samsung Galaxy S21 Ultra 5G",
        "series": "Galaxy S21 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s21_ultra_5g-10596.php"
    },
    {
        "model_name": "Samsung Galaxy S21+ 5G",
        "series": "Galaxy S21 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s21_plus_5g-10625.php"
    },
    {
        "model_name": "Samsung Galaxy S21 5G",
        "series": "Galaxy S21 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s21_5g-10626.php"
    },
    {
        "model_name": "Samsung Galaxy S21 FE 5G",
        "series": "Galaxy S21 Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_s21_fe_5g-10954.php"
    },
    {
        "model_name": "Samsung Galaxy A54 5G",
        "series": "Galaxy A Series",
        "url": "https://www.gsmarena.com/samsung_galaxy_a54-12070.php"
    }
]


class SamsungPhoneScraper:
    """Web scraper for extracting Samsung phone specifications from GSMArena."""

    def __init__(self, request_delay: float = 1.0, timeout: int = 15):
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch HTML content from a given URL with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = attempt * 5
                    logger.warning("Rate limit (429) hit. Backing off for %s seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.warning("HTTP %s while fetching %s", response.status_code, url)
            except requests.RequestException as exc:
                logger.warning("Attempt %d failed for %s: %s", attempt, url, exc)
                time.sleep(attempt * 2)
        return None

    def parse_tables(self, soup: BeautifulSoup) -> Dict[str, Dict[str, str]]:
        """Parse all specification tables into a clean nested dictionary."""
        specs_by_category: Dict[str, Dict[str, str]] = {}
        tables = soup.find_all("table")

        for table in tables:
            th = table.find("th")
            category = th.get_text().strip() if th else "General"
            if not category:
                continue

            if category not in specs_by_category:
                specs_by_category[category] = {}

            current_key = "details"
            for tr in table.find_all("tr"):
                ttl = tr.find("td", class_="ttl")
                nfo = tr.find("td", class_="nfo")
                if nfo:
                    raw_key = ttl.get_text().strip() if ttl else ""
                    if raw_key:
                        current_key = raw_key
                    # Clean the value
                    val = nfo.get_text("\n").strip()
                    # Clean up multiple newlines or non-breaking spaces
                    val = re.sub(r"\r", "", val)
                    val = re.sub(r"\n{2,}", "\n", val)
                    val = val.replace("\xa0", " ").strip()

                    if current_key in specs_by_category[category]:
                        specs_by_category[category][current_key] += "\n" + val
                    else:
                        specs_by_category[category][current_key] = val

        return specs_by_category

    def extract_display_features(self, display_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract display size and specifications."""
        size_raw = display_sec.get("Size", "")
        # Extract inches e.g. "6.8 inches" or '6.8"'
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inches|\")", size_raw, re.IGNORECASE)
        display_inches = f"{match.group(1)} inches" if match else (size_raw.split(",")[0].strip() if size_raw else "N/A")
        
        # Screen to body ratio
        ratio_match = re.search(r"~?(\d+(?:\.\d+)?%)\s*screen-to-body", size_raw)
        screen_to_body = ratio_match.group(1) if ratio_match else "N/A"

        return {
            "display_size": display_inches,
            "display_size_full": size_raw,
            "screen_to_body_ratio": screen_to_body,
            "display_type": display_sec.get("Type", "N/A"),
            "resolution": display_sec.get("Resolution", "N/A"),
            "protection": display_sec.get("Protection", "N/A")
        }

    def extract_camera_features(self, main_cam_sec: Dict[str, str], selfie_cam_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract camera specifications for both main and selfie cameras."""
        def parse_cam(sec: Dict[str, str], default_name: str) -> Dict[str, Any]:
            setup = default_name
            lenses: List[str] = []
            primary_mp = "N/A"

            for k, v in sec.items():
                if k.lower() not in ["features", "video"]:
                    setup = k
                    # Split lenses
                    lines = [line.strip() for line in v.split("\n") if line.strip()]
                    lenses.extend(lines)
                    # Try to find primary sensor megapixel
                    mp_match = re.search(r"(\d+)\s*MP", v)
                    if mp_match and primary_mp == "N/A":
                        primary_mp = f"{mp_match.group(1)} MP"

            return {
                "setup": setup,
                "primary_sensor": primary_mp,
                "lenses": lenses,
                "features": sec.get("Features", "N/A"),
                "video": sec.get("Video", "N/A")
            }

        return {
            "main_camera": parse_cam(main_cam_sec, "Main"),
            "selfie_camera": parse_cam(selfie_cam_sec, "Selfie")
        }

    def extract_battery_features(self, battery_sec: Dict[str, str], tests_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract battery capacity, charging, and endurance life scores."""
        battery_type_raw = battery_sec.get("Type", "")
        cap_match = re.search(r"(\d+)\s*mAh", battery_type_raw, re.IGNORECASE)
        capacity_str = f"{cap_match.group(1)} mAh" if cap_match else "N/A"
        capacity_numeric = int(cap_match.group(1)) if cap_match else None

        active_use = tests_sec.get("Battery", "N/A")
        endurance = tests_sec.get("Battery (old)", "N/A")

        return {
            "capacity": capacity_str,
            "capacity_numeric_mah": capacity_numeric,
            "battery_type": battery_type_raw if battery_type_raw else "N/A",
            "charging": battery_sec.get("Charging", "N/A"),
            "active_use_score": active_use,
            "endurance_rating": endurance
        }

    def extract_processor_features(self, platform_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract processor (chipset, CPU, GPU, OS) details."""
        chipset = platform_sec.get("Chipset", "N/A")
        cpu = platform_sec.get("CPU", "N/A")
        gpu = platform_sec.get("GPU", "N/A")
        os_version = platform_sec.get("OS", "N/A")

        return {
            "chipset": chipset,
            "cpu": cpu,
            "gpu": gpu,
            "operating_system": os_version
        }

    def extract_memory_features(self, memory_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract RAM, internal storage, and expandable storage."""
        return {
            "internal_storage_and_ram": memory_sec.get("Internal", "N/A"),
            "card_slot": memory_sec.get("Card slot", "N/A")
        }

    def extract_body_features(self, body_sec: Dict[str, str]) -> Dict[str, Any]:
        """Extract dimensions, weight, build materials, and IP rating."""
        sim_val = body_sec.get("SIM", "")
        ip_rating = "N/A"
        ip_match = re.search(r"(IP\d{2})", sim_val)
        if ip_match:
            ip_rating = ip_match.group(1)

        return {
            "dimensions": body_sec.get("Dimensions", "N/A"),
            "weight": body_sec.get("Weight", "N/A"),
            "build": body_sec.get("Build", "N/A"),
            "sim": sim_val,
            "ip_rating": ip_rating
        }

    def scrape_phone(self, phone_target: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Scrape all specifications and extracted key features for a single phone."""
        url = phone_target["url"]
        logger.info("Scraping %s from %s ...", phone_target.get("model_name", "phone"), url)

        html = self.fetch_page(url)
        if not html:
            logger.error("Failed to retrieve HTML for %s", url)
            return None

        soup = BeautifulSoup(html, "lxml")

        # Official title
        title_el = soup.find("h1", class_="specs-phone-name-title")
        model_name = title_el.get_text().strip() if title_el else phone_target.get("model_name", "Unknown Samsung Galaxy")

        # Image URL
        img_div = soup.find("div", class_="specs-photo-main")
        image_url = img_div.find("img").get("src") if (img_div and img_div.find("img")) else None

        # Parse all table specs
        all_specs = self.parse_tables(soup)

        # Extract targeted feature groups
        display_features = self.extract_display_features(all_specs.get("Display", {}))
        camera_features = self.extract_camera_features(
            all_specs.get("Main Camera", {}),
            all_specs.get("Selfie camera", {})
        )
        battery_features = self.extract_battery_features(
            all_specs.get("Battery", {}),
            all_specs.get("Tests", {})
        )
        processor_features = self.extract_processor_features(all_specs.get("Platform", {}))
        memory_features = self.extract_memory_features(all_specs.get("Memory", {}))
        body_features = self.extract_body_features(all_specs.get("Body", {}))
        launch_sec = all_specs.get("Launch", {})

        phone_record = {
            "model_name": model_name,
            "series": phone_target.get("series", "Galaxy Series"),
            "brand": "Samsung",
            "url": url,
            "image_url": image_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            # Primary Key Features requested by user
            "key_features": {
                "display_size": display_features["display_size"],
                "display": display_features,
                "processor": processor_features,
                "camera_specs": camera_features,
                "battery_life": battery_features
            },
            # Secondary contextual specs for query system
            "memory": memory_features,
            "body": body_features,
            "launch": {
                "announced": launch_sec.get("Announced", "N/A"),
                "status": launch_sec.get("Status", "N/A")
            },
            # Complete raw specs dictionary for full completeness
            "all_specifications": all_specs
        }

        return phone_record

    def scrape_all(self, target_phones: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """Scrape all target phones with courteous rate limiting."""
        targets = target_phones or DEFAULT_TARGET_PHONES
        results: List[Dict[str, Any]] = []

        total = len(targets)
        logger.info("Beginning scrape of %d Samsung Galaxy devices...", total)

        for idx, target in enumerate(targets, start=1):
            try:
                record = self.scrape_phone(target)
                if record:
                    results.append(record)
                    logger.info("[%d/%d] Successfully scraped: %s", idx, total, record["model_name"])
                else:
                    logger.warning("[%d/%d] Skipping failed target: %s", idx, total, target.get("model_name"))
            except Exception as e:
                logger.error("Error processing %s: %s", target.get("model_name"), e, exc_info=True)

            # Courteous delay between requests
            if idx < total:
                time.sleep(self.request_delay)

        logger.info("Scraping completed. Successfully scraped %d out of %d phones.", len(results), total)
        return results

    def save_to_json(self, phones_data: List[Dict[str, Any]], output_filepath: str) -> None:
        """Save scraped phones dataset to a cleanly indented JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        payload = {
            "metadata": {
                "system": "Samsung Phone Query and Review System",
                "step": 1,
                "description": "Scraped specifications for Samsung Galaxy phones",
                "source": "GSMArena",
                "total_phones": len(phones_data),
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "phones": phones_data
        }

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d phones to JSON at %s", len(phones_data), output_filepath)


def main():
    """Main execution function."""
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samsung_phones_data.json")
    scraper = SamsungPhoneScraper(request_delay=1.0)
    data = scraper.scrape_all()
    scraper.save_to_json(data, output_file)
    print(f"\nDone! Scraped {len(data)} phones into '{output_file}'.")


if __name__ == "__main__":
    main()
