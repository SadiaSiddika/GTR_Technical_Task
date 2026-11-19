"""
Technical Analyst Agent
=======================
Part of the Samsung Phone Query and Review System.

Role:
Specialized AI Agent that retrieves, analyzes, and synthesizes specific
technical data, hardware specifications, and engineering metrics directly
from the relational database (SQLAlchemy) and vector database (FAISS).

Capabilities:
1. Exact Hardware Lookup: Chipset nodes, CPU clusters, GPU, RAM, sensors.
2. Optics & Camera Analysis: Sensor sizes, micron pixel pitch, OIS, optical zoom, ISP.
3. Power Delivery & Battery: mAh capacities, wired/wireless wattage, active use endurance.
4. Display & Panel Analysis: Dynamic AMOLED 2X, peak nits, LTPO refresh rates, Gorilla Armor.
5. Multi-Device Technical Comparisons: Structured delta analysis between phone generations.
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

# Database and RAG dependencies
from db_manager import Base, CameraLens, Phone, PhoneSpecification, get_database_engine
from rag_pipeline import VectorStoreManager, query_phone_specs

# LangChain tool interfaces
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TechnicalAnalystAgent")


# ==============================================================================
# 1. Specialized Technical Analyst Tools
# ==============================================================================

class TechnicalDataRetriever:
    """Core database and vector retrieval tools used by the Technical Analyst Agent."""

    def __init__(self):
        self.engine, self.db_type = get_database_engine()
        self.vector_manager = VectorStoreManager()
        self.vector_manager.initialize_vector_store()

    def get_phone_by_name_fuzzy(self, session: Session, query_name: str) -> Optional[Phone]:
        """Fuzzy matches a phone model name from user query."""
        clean_q = query_name.lower().replace("samsung", "").replace("galaxy", "").strip()

        phones = session.query(Phone).all()

        # 1. Exact match after cleaning
        for p in phones:
            p_clean = p.model_name.lower().replace("samsung", "").replace("galaxy", "").strip()
            if clean_q == p_clean:
                return p

        # 2. Strict word boundary match (prefer exact model variants)
        for p in phones:
            p_clean = p.model_name.lower().replace("samsung", "").replace("galaxy", "").strip()
            # If query has 'ultra', match ultra; if query has 'fe', match fe; if query has '+' or 'plus', match plus
            has_ultra_q = "ultra" in clean_q
            has_ultra_p = "ultra" in p_clean
            has_plus_q = "+" in clean_q or "plus" in clean_q
            has_plus_p = "+" in p_clean or "plus" in p_clean
            has_fe_q = "fe" in clean_q
            has_fe_p = "fe" in p_clean

            if has_ultra_q == has_ultra_p and has_plus_q == has_plus_p and has_fe_q == has_fe_p:
                if re.search(r"\b" + re.escape(clean_q.split()[0]) + r"\b", p_clean):
                    return p

        # 3. Fallback token overlap
        q_tokens = set(clean_q.split())
        best_match = None
        best_score = 0
        for p in phones:
            p_tokens = set(p.model_name.lower().split())
            common = len(q_tokens.intersection(p_tokens))
            if common > best_score:
                best_score = common
                best_match = p

        return best_match if best_score >= 1 else None

    def fetch_full_technical_profile(self, model_query: str) -> Dict[str, Any]:
        """Retrieves comprehensive technical metrics from relational database."""
        with Session(self.engine) as session:
            phone = self.get_phone_by_name_fuzzy(session, model_query)
            if not phone:
                return {"error": f"Device matching '{model_query}' not found in database."}

            # Lenses
            main_lenses = [
                cl.lens_description for cl in phone.camera_lenses if cl.camera_type == "main"
            ]
            selfie_lenses = [
                cl.lens_description for cl in phone.camera_lenses if cl.camera_type == "selfie"
            ]

            # Categorized specifications
            specs_by_cat: Dict[str, Dict[str, str]] = {}
            for s in phone.specifications:
                if s.category not in specs_by_cat:
                    specs_by_cat[s.category] = {}
                specs_by_cat[s.category][s.spec_name] = s.spec_value

            return {
                "id": phone.id,
                "model_name": phone.model_name,
                "series": phone.series,
                "brand": phone.brand,
                "launch": {
                    "announced": phone.announced,
                    "status": phone.status
                },
                "display": {
                    "size": phone.display_size,
                    "size_numeric": phone.display_size_numeric,
                    "type": phone.display_type,
                    "resolution": phone.display_resolution,
                    "protection": phone.display_protection
                },
                "processor": {
                    "chipset": phone.chipset,
                    "cpu": phone.cpu,
                    "gpu": phone.gpu,
                    "os": phone.operating_system
                },
                "camera": {
                    "main_setup": phone.main_camera_setup,
                    "main_primary_sensor": phone.main_camera_primary,
                    "main_lenses": main_lenses,
                    "main_video": phone.main_camera_video,
                    "selfie_setup": phone.selfie_camera_setup,
                    "selfie_primary_sensor": phone.selfie_camera_primary,
                    "selfie_lenses": selfie_lenses,
                    "selfie_video": phone.selfie_camera_video
                },
                "battery": {
                    "capacity": phone.battery_capacity,
                    "capacity_mah": phone.battery_capacity_mah,
                    "type": phone.battery_type,
                    "charging": phone.charging,
                    "active_use_score": phone.active_use_score,
                    "endurance_rating": phone.endurance_rating
                },
                "build": {
                    "dimensions": phone.dimensions,
                    "weight": phone.weight,
                    "ip_rating": phone.ip_rating,
                    "memory": phone.internal_storage_and_ram
                },
                "raw_categories": specs_by_cat
            }

    def compare_two_devices(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """Performs a direct technical comparison between two devices."""
        profile_a = self.fetch_full_technical_profile(model_a)
        profile_b = self.fetch_full_technical_profile(model_b)

        if "error" in profile_a:
            return profile_a
        if "error" in profile_b:
            return profile_b

        return {
            "device_a": profile_a,
            "device_b": profile_b,
            "comparison_matrix": {
                "Display Size": (profile_a["display"]["size"], profile_b["display"]["size"]),
                "Display Type": (profile_a["display"]["type"], profile_b["display"]["type"]),
                "Screen Resolution": (profile_a["display"]["resolution"], profile_b["display"]["resolution"]),
                "Processor / SoC": (profile_a["processor"]["chipset"], profile_b["processor"]["chipset"]),
                "GPU": (profile_a["processor"]["gpu"], profile_b["processor"]["gpu"]),
                "Main Primary Sensor": (profile_a["camera"]["main_primary_sensor"], profile_b["camera"]["main_primary_sensor"]),
                "Main Camera Setup": (profile_a["camera"]["main_setup"], profile_b["camera"]["main_setup"]),
                "Battery Capacity": (profile_a["battery"]["capacity"], profile_b["battery"]["capacity"]),
                "Charging Speed": (profile_a["battery"]["charging"].split("\n")[0] if profile_a["battery"]["charging"] else "N/A",
                                    profile_b["battery"]["charging"].split("\n")[0] if profile_b["battery"]["charging"] else "N/A"),
                "Weight": (profile_a["build"]["weight"], profile_b["build"]["weight"]),
                "IP Rating": (profile_a["build"]["ip_rating"], profile_b["build"]["ip_rating"])
            }
        }

    def filter_by_criteria(
        self,
        min_battery: Optional[int] = None,
        chipset_kw: Optional[str] = None,
        min_camera_mp: Optional[int] = None,
        series_kw: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter devices from relational database based on strict hardware parameters."""
        with Session(self.engine) as session:
            query = session.query(Phone)

            if min_battery:
                query = query.filter(Phone.battery_capacity_mah >= min_battery)
            if chipset_kw:
                query = query.filter(Phone.chipset.ilike(f"%{chipset_kw}%"))
            if series_kw:
                query = query.filter(Phone.series.ilike(f"%{series_kw}%"))

            phones = query.all()
            results = []

            for p in phones:
                # Check camera MP if specified
                if min_camera_mp:
                    prim_str = p.main_camera_primary or ""
                    mp_match = re.search(r"(\d+)\s*mp", prim_str, re.IGNORECASE)
                    if not mp_match or int(mp_match.group(1)) < min_camera_mp:
                        continue

                results.append({
                    "model_name": p.model_name,
                    "series": p.series,
                    "display": p.display_size,
                    "chipset": p.chipset,
                    "primary_camera": p.main_camera_primary,
                    "battery": p.battery_capacity,
                    "charging": p.charging.split("\n")[0] if p.charging else "N/A"
                })

            return results


# ==============================================================================
# 2. Technical Analyst Agent Implementation
# ==============================================================================

class TechnicalAnalystAgent:
    """
    Autonomous Technical Analyst Agent for Samsung Galaxy devices.
    Processes user inquiries, invokes database & vector tools, and synthesizes
    structured, engineering-grade technical analyses.
    """

    def __init__(self):
        self.retriever = TechnicalDataRetriever()
        logger.info("Technical Analyst Agent initialized with Database (%s) & FAISS vector tools.", self.retriever.db_type)

    def analyze(self, user_query: str) -> Dict[str, Any]:
        """
        Main entrypoint: analyzes user query, retrieves relevant technical data,
        and generates a rigorous technical analysis report.
        """
        logger.info("Technical Analyst Agent received query: '%s'", user_query)
        q_lower = user_query.lower()

        # Step 1: Detect Intent & Target Entities
        detected_models = self._extract_target_models(q_lower)
        is_comparison = len(detected_models) >= 2 or any(kw in q_lower for kw in ["compare", "vs", "versus", "difference", "between"])
        is_filter_query = any(kw in q_lower for kw in ["which phones", "list phones", "phones with", "show phones", "find phones", "filter"])

        report = ""
        structured_data: Dict[str, Any] = {}
        sources: List[str] = []

        # Route 1: Multi-Device Comparison Analysis
        if is_comparison and len(detected_models) >= 2:
            model_a, model_b = detected_models[0], detected_models[1]
            cmp_data = self.retriever.compare_two_devices(model_a, model_b)
            if "error" not in cmp_data:
                structured_data = cmp_data
                sources = [cmp_data["device_a"]["model_name"], cmp_data["device_b"]["model_name"]]
                report = self._generate_comparison_report(user_query, cmp_data)
            else:
                report = f"⚠️ {cmp_data.get('error')}"

        # Route 2: Single Device Deep Technical Profile
        elif detected_models and not is_filter_query:
            target_model = detected_models[0]
            profile = self.retriever.fetch_full_technical_profile(target_model)
            if "error" not in profile:
                structured_data = profile
                sources = [profile["model_name"]]
                report = self._generate_single_device_report(user_query, profile)
            else:
                # Fallback to RAG vector search if exact model name is ambiguous
                rag_res = query_phone_specs(user_query)
                report = rag_res.get("answer", "")
                sources = rag_res.get("source_phones", [])

        # Route 3: Parametric Specification Filter
        elif is_filter_query or any(kw in q_lower for kw in ["5000", "snapdragon", "exynos", "200 mp", "108 mp"]):
            min_battery = 5000 if ("5000" in q_lower or "5000 mah" in q_lower) else None
            min_camera = 200 if "200" in q_lower else (108 if "108" in q_lower else None)
            chipset = "snapdragon" if "snapdragon" in q_lower else ("exynos" if "exynos" in q_lower else None)

            filtered = self.retriever.filter_by_criteria(
                min_battery=min_battery,
                chipset_kw=chipset,
                min_camera_mp=min_camera
            )
            structured_data = {"matches": filtered, "count": len(filtered)}
            sources = [p["model_name"] for p in filtered]
            report = self._generate_filter_report(user_query, filtered)

        # Route 4: Fallback to Hybrid RAG Vector Retrieval
        else:
            rag_res = query_phone_specs(user_query)
            report = rag_res.get("answer", "")
            sources = rag_res.get("source_phones", [])
            structured_data = {"rag_matches": rag_res.get("retrieved_matches", [])}

        return {
            "agent": "Technical Analyst Agent",
            "query": user_query,
            "report": report,
            "sources": sources,
            "structured_data": structured_data
        }

    def _extract_target_models(self, q_lower: str) -> List[str]:
        """Identifies which Samsung models are referenced in the query."""
        all_known = [
            ("s24 ultra", "Samsung Galaxy S24 Ultra"),
            ("s24+", "Samsung Galaxy S24+"),
            ("s24 plus", "Samsung Galaxy S24+"),
            ("s24", "Samsung Galaxy S24"),
            ("s23 ultra", "Samsung Galaxy S23 Ultra"),
            ("s23+", "Samsung Galaxy S23+"),
            ("s23 plus", "Samsung Galaxy S23+"),
            ("s23 fe", "Samsung Galaxy S23 FE"),
            ("s23", "Samsung Galaxy S23"),
            ("s22 ultra", "Samsung Galaxy S22 Ultra 5G"),
            ("s22+", "Samsung Galaxy S22+ 5G"),
            ("s22 plus", "Samsung Galaxy S22+ 5G"),
            ("s22", "Samsung Galaxy S22 5G"),
            ("s21 ultra", "Samsung Galaxy S21 Ultra 5G"),
            ("s21+", "Samsung Galaxy S21+ 5G"),
            ("s21 plus", "Samsung Galaxy S21+ 5G"),
            ("s21 fe", "Samsung Galaxy S21 FE 5G"),
            ("s21", "Samsung Galaxy S21 5G"),
            ("a54", "Samsung Galaxy A54 5G")
        ]

        found = []
        for token, full_name in all_known:
            # Word boundary search
            pattern = r"\b" + re.escape(token) + r"\b"
            if re.search(pattern, q_lower):
                if full_name not in found:
                    found.append(full_name)

        return found

    def _generate_single_device_report(self, query: str, p: Dict[str, Any]) -> str:
        """Constructs an engineering-grade technical analysis report for a single device."""
        q_lower = query.lower()
        title = f"🔬 Technical Analysis Report: {p['model_name']}"

        sections = [f"## {title}\n"]

        # 1. Executive Summary
        sections.append(
            f"**Device Identity:** {p['model_name']} ({p['series']})\n"
            f"**Release Status:** {p['launch']['announced']} • {p['launch']['status']}\n"
            f"**Form Factor:** {p['build']['dimensions']} | **Weight:** {p['build']['weight']} | **Ingress Protection:** {p['build']['ip_rating']}\n"
        )

        # 2. Camera System & Sensor Architecture
        if any(kw in q_lower for kw in ["camera", "sensor", "lens", "photo", "video", "mp"]) or not any(kw in q_lower for kw in ["battery", "processor", "display", "screen"]):
            cam = p["camera"]
            lenses_fmt = "\n".join([f"  • {l}" for l in cam["main_lenses"]]) if cam["main_lenses"] else "  • Standard primary sensor"
            selfie_fmt = "\n".join([f"  • {l}" for l in cam["selfie_lenses"]]) if cam["selfie_lenses"] else "  • Standard selfie sensor"

            sections.append(
                f"### 📸 Camera Subsystem Architecture\n"
                f"- **Configuration:** {cam['main_setup']} Rear Camera Module\n"
                f"- **Primary Sensor:** {cam['main_primary_sensor']}\n"
                f"- **Detailed Optical Pipeline & Sensor Matrix:**\n{lenses_fmt}\n"
                f"- **Video Encoding Capabilities:** {cam['main_video']}\n"
                f"- **Front Selfie Optics:** {cam['selfie_setup']} ({cam['selfie_primary_sensor']})\n{selfie_fmt}\n"
                f"- **Selfie Video Encoding:** {cam['selfie_video']}\n"
            )

        # 3. Silicon & Processing Architecture
        if any(kw in q_lower for kw in ["processor", "chipset", "cpu", "gpu", "soc", "performance", "platform"]) or not any(kw in q_lower for kw in ["battery", "camera", "display"]):
            proc = p["processor"]
            sections.append(
                f"### ⚡ Compute & Platform Architecture\n"
                f"- **SoC / Chipset:** {proc['chipset']}\n"
                f"- **CPU Core Topology & Frequencies:** {proc['cpu']}\n"
                f"- **GPU Graphics Processing Unit:** {proc['gpu']}\n"
                f"- **OS & Software Longevity:** {proc['os']}\n"
                f"- **RAM & Memory Configurations:** {p['build']['memory']}\n"
            )

        # 4. Display Subsystem
        if any(kw in q_lower for kw in ["display", "screen", "panel", "resolution", "nits", "amoled", "inch"]) or not any(kw in q_lower for kw in ["battery", "camera", "processor"]):
            disp = p["display"]
            sections.append(
                f"### 📱 Display Panel & Optical Characteristics\n"
                f"- **Panel Dimension:** {disp['size']}\n"
                f"- **Panel Technology:** {disp['type']}\n"
                f"- **Resolution & Pixel Density:** {disp['resolution']}\n"
                f"- **Cover Glass Substrate:** {disp['protection']}\n"
            )

        # 5. Power & Thermal Delivery
        if any(kw in q_lower for kw in ["battery", "charging", "mah", "watt", "power", "endurance"]) or not any(kw in q_lower for kw in ["camera", "processor", "display"]):
            batt = p["battery"]
            sections.append(
                f"### 🔋 Power Delivery & Battery Subsystem\n"
                f"- **Cell Capacity:** {batt['capacity']} ({batt['type']})\n"
                f"- **Charging Protocol & Throughput:** {batt['charging'].replace(chr(10), ', ') if batt['charging'] else 'N/A'}\n"
                f"- **Active Use Battery Test Score:** {batt['active_use_score']}\n"
                f"- **Endurance Rating:** {batt['endurance_rating']}\n"
            )

        # 6. Technical Analyst Verdict
        sections.append(
            f"### 💡 Technical Analyst Verdict\n"
            f"The **{p['model_name']}** integrates high-density engineering with a "
            f"**{p['display']['size']}** display, **{p['processor']['chipset'].split('(')[0].strip() if p['processor']['chipset'] else 'flagship'}** SoC, and a **{p['camera']['main_primary_sensor']}** camera pipeline. "
            f"Rated **{p['build']['ip_rating']}** for chassis integrity with **{p['battery']['capacity']}** capacity."
        )

        return "\n".join(sections)

    def _generate_comparison_report(self, query: str, cmp_data: Dict[str, Any]) -> str:
        """Constructs an engineering delta comparison report between two devices."""
        dev_a = cmp_data["device_a"]
        dev_b = cmp_data["device_b"]
        matrix = cmp_data["comparison_matrix"]

        table_rows = []
        for metric, (val_a, val_b) in matrix.items():
            clean_a = str(val_a).replace("\n", " ").strip()
            clean_b = str(val_b).replace("\n", " ").strip()
            table_rows.append(f"| **{metric}** | {clean_a} | {clean_b} |")

        table_md = (
            f"| Technical Parameter | {dev_a['model_name']} | {dev_b['model_name']} |\n"
            f"|---|---|---|\n" + "\n".join(table_rows)
        )

        return (
            f"## ⚖️ Technical Comparative Analysis: {dev_a['model_name']} vs {dev_b['model_name']}\n\n"
            f"### 📊 Hardware Specification Matrix\n\n"
            f"{table_md}\n\n"
            f"### 🔬 Key Architectural Differences & Engineering Highlights\n"
            f"1. **Silicon Compute:** {dev_a['model_name']} runs on `{dev_a['processor']['chipset'].split('(')[0].strip()}`, whereas {dev_b['model_name']} is powered by `{dev_b['processor']['chipset'].split('(')[0].strip()}`.\n"
            f"2. **Optics Pipeline:** {dev_a['model_name']} features a `{dev_a['camera']['main_setup']} ({dev_a['camera']['main_primary_sensor']})` system, compared to `{dev_b['camera']['main_setup']} ({dev_b['camera']['main_primary_sensor']})` on {dev_b['model_name']}.\n"
            f"3. **Power Architecture:** {dev_a['model_name']} is equipped with `{dev_a['battery']['capacity']}` vs `{dev_b['battery']['capacity']}` on {dev_b['model_name']}.\n\n"
            f"### 💡 Engineering Summary\n"
            f"Both devices reflect their respective generation's flagship targets. {dev_b['model_name']} offers progressive architectural refinement in silicon node efficiency and optical processing over {dev_a['model_name']}."
        )

    def _generate_filter_report(self, query: str, matches: List[Dict[str, Any]]) -> str:
        """Constructs a report listing devices that match criteria."""
        if not matches:
            return "No Samsung Galaxy models in the database matched the specified technical criteria."

        rows = []
        for idx, m in enumerate(matches, 1):
            chip = m['chipset'].split('(')[0].strip() if m['chipset'] else 'N/A'
            rows.append(f"| {idx} | **{m['model_name']}** | {m['display']} | `{chip}` | {m['primary_camera']} | **{m['battery']}** ({m['charging']}) |")

        table = (
            f"| # | Model Name | Display | Chipset | Main Sensor | Battery & Charging |\n"
            f"|---|---|---|---|---|---|\n" + "\n".join(rows)
        )

        return (
            f"## 📋 Technical Query Matches ({len(matches)} Devices Found)\n\n"
            f"Based on your query: *\"{query}\"*, the following devices were retrieved from the relational database:\n\n"
            f"{table}\n\n"
            f"💡 *All specifications are verified from GSMArena and stored in the local relational database.*"
        )


# Global singleton agent instance
_global_analyst_agent: Optional[TechnicalAnalystAgent] = None


def get_technical_analyst_agent() -> TechnicalAnalystAgent:
    """Returns the singleton TechnicalAnalystAgent instance."""
    global _global_analyst_agent
    if _global_analyst_agent is None:
        _global_analyst_agent = TechnicalAnalystAgent()
    return _global_analyst_agent


# ==============================================================================
# 3. CLI Demonstration & Benchmarking
# ==============================================================================

def main():
    print("=" * 85)
    print(" Technical Analyst Agent - Samsung Phone Query & Review System")
    print("=" * 85)

    agent = get_technical_analyst_agent()

    test_queries = [
        "What are the camera specs and sensor details of the Samsung Galaxy S23?",
        "Compare the technical specifications between Samsung Galaxy S23 and S24 Ultra",
        "Which Samsung Galaxy phones have 5000 mAh battery?",
        "Give me a technical breakdown of the processor and GPU in the Samsung Galaxy S22 Ultra"
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{'='*35} TEST QUERY {idx} {'='*35}")
        print(f"User Query: {q}")
        print("-" * 80)

        result = agent.analyze(q)

        print(f"Agent: {result['agent']}")
        print(f"Sources Cited: {', '.join(result['sources'])}")
        print("\n" + result["report"])
        print("-" * 80)


if __name__ == "__main__":
    main()
