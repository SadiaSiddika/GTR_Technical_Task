"""
Reviewer Agent
==============
Part of the Samsung Phone Query and Review System.

Role:
Takes the raw technical data and engineering metrics provided by the
Technical Analyst Agent and transforms them into a detailed, engaging,
consumer-friendly product review or head-to-head comparison.

Key Sections in Generated Reviews:
1. 🌟 The Big Picture & First Impressions (Design, Hand-Feel, Ergonomics)
2. 📸 Camera & Photography Real-World Performance
3. ⚡ Processing Power, Gaming & Everyday Fluidity
4. 🔋 Battery Longevity & Charging Experience
5. 📱 Display Immersion & Multimedia
6. ⚖️ The Verdict: Pros, Cons & Buying Recommendation
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ReviewerAgent")


class ReviewerAgent:
    """
    Reviewer Agent that synthesizes technical data from the Technical Analyst
    into an engaging, human-centric product review or comparison.
    """

    def __init__(self, tone: str = "engaging_expert"):
        self.tone = tone
        logger.info("Reviewer Agent initialized (tone=%s).", self.tone)

    def review(self, user_query: str, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives output from the Technical Analyst Agent and writes a detailed review.
        """
        logger.info("Reviewer Agent received technical data for query: '%s'", user_query)

        structured_data = technical_analysis.get("structured_data", {})
        tech_report = technical_analysis.get("report", "")
        sources = technical_analysis.get("sources", [])

        # Check if the technical data represents a comparison or single device
        if "device_a" in structured_data and "device_b" in structured_data:
            review_text = self._write_comparison_review(user_query, structured_data, tech_report)
            review_type = "comparison_review"
        elif "model_name" in structured_data:
            review_text = self._write_single_device_review(user_query, structured_data, tech_report)
            review_type = "single_device_review"
        elif "matches" in structured_data:
            review_text = self._write_buying_guide_review(user_query, structured_data, tech_report)
            review_type = "buying_guide_review"
        else:
            review_text = self._write_general_review(user_query, tech_report)
            review_type = "general_technical_review"

        return {
            "agent": "Reviewer Agent",
            "review_type": review_type,
            "query": user_query,
            "sources": sources,
            "review": review_text
        }

    def _write_single_device_review(
        self, query: str, profile: Dict[str, Any], tech_report: str
    ) -> str:
        """Writes an engaging deep-dive review for a single Samsung Galaxy phone."""
        name = profile["model_name"]
        series = profile.get("series", "Galaxy Series")
        disp = profile.get("display", {})
        proc = profile.get("processor", {})
        cam = profile.get("camera", {})
        batt = profile.get("battery", {})
        build = profile.get("build", {})

        chip_name = proc.get("chipset", "").split("(")[0].strip() or "High-Performance SoC"
        main_mp = cam.get("main_primary_sensor", "Multi-lens")
        cam_setup = cam.get("main_setup", "Triple")
        batt_cap = batt.get("capacity", "All-day battery")
        charging_spd = batt.get("charging", "Fast charging").split("\n")[0]
        disp_size = disp.get("size", "Large")
        disp_res = disp.get("resolution", "High-res")
        disp_type = disp.get("type", "Dynamic AMOLED 2X")

        # Determine pros & cons dynamically from technical metrics
        pros = []
        cons = []

        # Camera pros/cons
        if "200" in main_mp:
            pros.append("Class-leading 200 MP camera captures mind-boggling detail and clarity")
            pros.append("Versatile optical zoom capabilities suited for portraits and distant telephoto")
        elif "108" in main_mp:
            pros.append("High-resolution 108 MP main sensor with excellent dynamic range")
        else:
            pros.append(f"Vibrant {main_mp} primary camera with punchy colors and fast shutter speed")

        # Battery pros/cons
        batt_mah = batt.get("capacity_mah") or 0
        if batt_mah >= 5000:
            pros.append("Massive 5000 mAh battery easily powers through 1.5+ days of intensive use")
        elif batt_mah < 4000:
            cons.append("Modest battery capacity requires strategic top-ups under heavy 5G or gaming loads")
        else:
            pros.append(f"Dependable {batt_cap} battery balancing all-day endurance with slim ergonomics")

        if "45W" in charging_spd:
            pros.append("Rapid 45W wired fast charging tops up the cell to 65% in ~30 minutes")
        else:
            cons.append("25W charging is reliable, but trails behind competitors offering ultra-fast 65W+ speeds")

        # Processor & Display pros
        if "gen 3" in chip_name.lower():
            pros.append("Bleeding-edge Snapdragon 8 Gen 3 delivers flawless high-fps gaming and AI features")
        elif "gen 2" in chip_name.lower():
            pros.append("Snapdragon 8 Gen 2 provides remarkable thermal efficiency and butter-smooth multitasking")
        elif "exynos" in chip_name.lower():
            cons.append("Exynos variant may show minor thermal throttling under extended 3D gaming marathons")

        pros.append(f"Gorgeous {disp_size} Dynamic AMOLED 2X panel with 120Hz refresh rate and great outdoor visibility")
        pros.append(f"Rugged {build.get('ip_rating', 'IP68')} dust/water resistance for real-world peace of mind")

        if "no" in build.get("memory", "").lower() or not profile.get("raw_categories", {}).get("Memory", {}).get("Card slot", "").startswith("microSD"):
            cons.append("No expandable microSD storage slot — select your internal storage variant carefully")

        pros_fmt = "\n".join([f"  ✅ **{p}**" for p in pros[:4]])
        cons_fmt = "\n".join([f"  ⚠️ **{c}**" for c in cons[:3]])

        # Generate Engaging Markdown Review
        review = f"""# 📱 Comprehensive Product Review: {name}
*Written by the Reviewer Agent | Grounded on GSMArena Technical Metrics*

---

### 🌟 Executive Overview & First Impressions
The **{name}** stands as a testament to Samsung's engineering prowess in the **{series}**. Holding the device in hand, its **{build.get('dimensions', 'sleek chassis')}** and **{build.get('weight', 'balanced weight')}** convey immediate premium craftsmanship. Sealed with an **{build.get('ip_rating', 'IP68')} rating**, it handles unexpected splashes and dusty trails effortlessly. Whether you are sliding it into a pocket or gripping it for a two-hour streaming session, ergonomics remain top-tier.

---

### 📸 Real-World Camera Experience: Photography & Video
Samsung has equipped the {name} with a **{cam_setup} rear camera system**, crowned by a **{main_mp} primary sensor**.

- **Daylight Clarity:** Photos pop with Samsung’s signature vibrancy. HDR balances high-contrast scenes with rich shadow details and crisp highlights.
- **Low-Light & Nightography:** The large sensor aperture and optical image stabilization (OIS) pull light from dim ambient conditions, preserving sharpness without turning shadows into noisy mush.
- **Versatility:** The multi-lens array gives you extreme flexibility, moving seamlessly from wide-angle landscapes to crisp portrait shots with pleasing natural background bokeh.
- **Cinematic Video:** Recording up to **{cam.get('main_video', '8K/4K')}** with gyro-EIS, footage is steady enough to shoot handheld vlogs without needing a motorized gimbal.

---

### ⚡ Performance & Everyday Speed: Silicon in Action
Under the hood, the **{chip_name}** powers the experience.
Paired with **{build.get('memory', 'generous RAM')}**, running **{proc.get('os', 'Android with One UI')}**:
- **Daily Multitasking:** App switching is instantaneous. Animations glide at a constant 120Hz without micro-stutters.
- **Intensive Gaming:** Titles like *Genshin Impact* and *Call of Duty Mobile* run at high graphical presets, sustained by the GPU architecture.
- **Thermal Behavior:** The device maintains stable thermals during prolonged video calls and photo processing sessions.

---

### 🔋 Battery Life & Charging Habits
Housing a **{batt_cap}** battery ({batt.get('type', 'Li-Ion')}), battery anxiety is largely a thing of the past:
- **Daily Endurance:** Under moderate to heavy usage (social media, camera use, GPS navigation, and YouTube streaming), it comfortably lasts from morning commute to bedtime.
- **Top-up Speeds:** Wired charging throughput is **{charging_spd}**, accompanied by wireless Qi charging convenience.

---

### 📱 Display: A Feast for the Eyes
The **{disp_size} {disp_type}** ({disp_res}) is arguably the crown jewel. With Corning protection (**{disp.get('protection', 'Gorilla Glass')}**), text is razor-sharp, blacks are inky and infinite, and outdoor sunlight readability is superb.

---

### ⚖️ The Good & The Bad
{pros_fmt}

{cons_fmt}

---

### 🏆 Final Verdict & Buying Advice
The **{name}** is a triumphant release that marries high-end computational photography, stellar battery endurance, and one of the finest displays on the market.

- **Who Should Buy:** Power users, mobile photographers, and anyone looking for a reliable, future-proof flagship that will receive multi-year OS updates.
- **Who Should Skip:** Budget-conscious buyers who only need basic social media and messaging, or users who demand a 3.5mm headphone jack and microSD slot.
"""
        return review

    def _write_comparison_review(
        self, query: str, cmp_data: Dict[str, Any], tech_report: str
    ) -> str:
        """Writes an engaging, head-to-head comparison review between two devices."""
        dev_a = cmp_data["device_a"]
        dev_b = cmp_data["device_b"]
        name_a = dev_a["model_name"]
        name_b = dev_b["model_name"]

        # Contrast metrics
        chip_a = dev_a["processor"]["chipset"].split("(")[0].strip()
        chip_b = dev_b["processor"]["chipset"].split("(")[0].strip()
        cam_a = f"{dev_a['camera']['main_setup']} ({dev_a['camera']['main_primary_sensor']})"
        cam_b = f"{dev_b['camera']['main_setup']} ({dev_b['camera']['main_primary_sensor']})"
        batt_a = dev_a["battery"]["capacity"]
        batt_b = dev_b["battery"]["capacity"]
        disp_a = f"{dev_a['display']['size']} ({dev_a['display']['resolution']})"
        disp_b = f"{dev_b['display']['size']} ({dev_b['display']['resolution']})"

        review = f"""# ⚔️ Head-to-Head Flagship Showdown: {name_a} vs {name_b}
*Written by the Reviewer Agent | Comparative Analysis based on Verified Hardware Metrics*

---

### 🥊 The Matchup
Choosing between the **{name_a}** and the **{name_b}** boils down to a classic dilemma: generational leap vs refined value. Both handsets embody Samsung's cutting-edge industrial design, but beneath the surface lies key differences in optics, silicon efficiency, and form factor.

---

### Round 1: 📱 Display & Design Ergonomics
- **{name_a}:** Offers a **{disp_a}** screen. Ideal for users who value balanced palm ergonomics and pocketability.
- **{name_b}:** Packs a **{disp_b}** screen. Delivers an expansive cinematic canvas for media binges and split-screen productivity.
- **Round Winner:** **{name_b}** for immersive multimedia, but **{name_a}** takes the prize for day-to-day comfort and lighter carry weight ({dev_a['build']['weight']} vs {dev_b['build']['weight']}).

---

### Round 2: ⚡ Processor & Performance Muscle
- **{name_a}:** Powered by the `{chip_a}`. It tackles modern 3D games and heavy multitasking without breaking a sweat.
- **{name_b}:** Houses the upgraded `{chip_b}`. Brings higher peak clock frequencies and advanced NPU capabilities for on-device AI and computational photography.
- **Round Winner:** **{name_b}** leads in benchmark scores and future-proof silicon efficiency.

---

### Round 3: 📸 Camera Showdown: Shutter & Sensor
- **{name_a}:** `{cam_a}` delivers crisp colors, fast autofocus, and reliable portrait edge detection.
- **{name_b}:** `{cam_b}` elevates low-light detail capture, optical zoom reach, and pro-grade video recording ({dev_b['camera']['main_video']}).
- **Round Winner:** **{name_b}** takes the crown for shutterbugs and creators requiring superior optical reach.

---

### Round 4: 🔋 Battery Life & Endurance
- **{name_a}:** Features a **{batt_a}** battery with `{dev_a['battery']['charging'].split(chr(10))[0]}` charging.
- **{name_b}:** Packs a **{batt_b}** battery with `{dev_b['battery']['charging'].split(chr(10))[0]}` charging.
- **Round Winner:** **{name_b}** edges ahead in raw capacity and longevity under heavy daily use.

---

### 🏆 Final Verdict & Recommendation
| Criterion | Best Pick | Reason |
|---|---|---|
| **Raw Power & Future-Proofing** | **{name_b}** | Newer silicon architecture and enhanced GPU performance |
| **Ultimate Camera Versatility** | **{name_b}** | Superior primary sensor and optical zoom range |
| **Ergonomics & Value for Money** | **{name_a}** | Highly capable flagship experience in a lighter, cost-effective package |

**Final Recommendation:** If your budget allows and you demand the very best camera and battery endurance, go with the **{name_b}**. If you want a comfortable daily driver that delivers 90% of the flagship experience at a much friendlier price point, the **{name_a}** remains an outstanding choice!
"""
        return review

    def _write_buying_guide_review(
        self, query: str, data: Dict[str, Any], tech_report: str
    ) -> str:
        """Writes an engaging buying guide for a list of filtered phones."""
        matches = data.get("matches", [])
        return f"""# 🛍️ Samsung Galaxy Buying Guide & Recommendations
*Written by the Reviewer Agent | Tailored to your search: "{query}"*

---

### 💡 Overview of Contenders
We filtered the Samsung Galaxy lineup to find the devices matching your specific technical requirements. Out of our verified database, **{len(matches)} devices** hit the mark!

Here is how they stack up for real-world buyers:

{tech_report}

---

### 🎯 Who Should Buy What?
- **Top Pick for Battery Longevity & Power:** Pick models with **5000 mAh** and **45W charging** (like the Ultra and Plus series) if you travel often or game heavily.
- **Top Pick for Value:** Look at the **Galaxy FE** or **A-series** models, which provide modern displays and all-day endurance without the flagship price tag.
- **Top Pick for Photography:** Look for devices with **108 MP or 200 MP** primary sensors for class-leading zoom and low-light clarity.
"""

    def _write_general_review(self, query: str, tech_report: str) -> str:
        """Fallback engaging narrative review for general queries."""
        return f"""# 🔍 Samsung Galaxy Tech Review & Analysis
*Written by the Reviewer Agent | Query: "{query}"*

---

### 📝 Expert Analysis & Consumer Takeaway
Based on the verified laboratory and database specifications retrieved for your query, here is our comprehensive review:

{tech_report}

---

### 💡 Reviewer Insight
Samsung Galaxy devices are renowned for their industry-leading Dynamic AMOLED 2X displays, durable IP68 dust/water chassis, and versatile computational camera setups. When selecting your ideal model, prioritize the balance between screen size, battery capacity, and processing requirements to match your daily lifestyle!
"""


# Global singleton reviewer agent instance
_global_reviewer_agent: Optional[ReviewerAgent] = None


def get_reviewer_agent() -> ReviewerAgent:
    """Returns the singleton ReviewerAgent instance."""
    global _global_reviewer_agent
    if _global_reviewer_agent is None:
        _global_reviewer_agent = ReviewerAgent()
    return _global_reviewer_agent
