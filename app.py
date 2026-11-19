"""
Samsung Phone Query and Review System - Web Application
======================================================
Serves interactive frontend interface and REST APIs on port 4005.
"""

import json
import os
from flask import Flask, jsonify, render_template, request, send_file
import requests
from flask_cors import CORS
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from db_manager import Base, CameraLens, Phone, PhoneSpecification, get_database_engine, SamsungDataIngester, log_chat_query

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samsung_phones_data.json")

# Initialize relational database (PostgreSQL primary with graceful SQLite fallback)
db_engine, active_db_type = get_database_engine()
ingester = SamsungDataIngester(db_engine)
ingester.init_tables()
if os.path.exists(DATA_PATH):
    ingester.ingest_from_json(DATA_PATH)
DbSession = sessionmaker(bind=db_engine)


def load_data():
    if not os.path.exists(DATA_PATH):
        return {"metadata": {}, "phones": []}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    """Render the main frontend dashboard."""
    return render_template("index.html")


@app.route("/api/db/status", methods=["GET"])
def get_db_status():
    """Return database connection status and relational table statistics."""
    with DbSession() as session:
        phones_count = session.query(func.count(Phone.id)).scalar() or 0
        specs_count = session.query(func.count(PhoneSpecification.id)).scalar() or 0
        lenses_count = session.query(func.count(CameraLens.id)).scalar() or 0

    return jsonify({
        "status": "connected",
        "active_database": active_db_type,
        "tables": {
            "phones": phones_count,
            "phone_specifications": specs_count,
            "camera_lenses": lenses_count
        }
    })


@app.route("/api/rag/query", methods=["GET", "POST"])
def rag_query():
    """RAG-powered natural language question answering over phone specs."""
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        q = payload.get("query", "").strip()
    else:
        q = request.args.get("q", "").strip()

    if not q:
        return jsonify({"error": "Query parameter 'q' or json 'query' is required."}), 400

    # 1. Forward to FastAPI backend on port 8000
    try:
        r = requests.post("http://127.0.0.1:8000/query", json={"question": q, "force_route": "rag"}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return jsonify({
                "query": q,
                "answer": data.get("response", ""),
                "source_phones": data.get("source_phones", []),
                "retrieved_matches": data.get("metadata", {}).get("rag_matches", [])
            })
    except Exception:
        pass

    # 2. Local fallback if FastAPI server is starting or offline
    try:
        from rag_pipeline import query_phone_specs
        result = query_phone_specs(q)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Error running RAG query: {str(exc)}"}), 500


@app.route("/api/workflow/review", methods=["GET", "POST"])
def workflow_review():
    """Sequential Multi-Agent Workflow: Technical Analyst -> Reviewer Agent."""
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        q = payload.get("query", "").strip()
    else:
        q = request.args.get("q", "").strip()

    if not q:
        return jsonify({"error": "Query parameter 'q' or json 'query' is required."}), 400

    # 1. Forward to FastAPI backend on port 8000
    try:
        r = requests.post("http://127.0.0.1:8000/query", json={"question": q, "force_route": "crewai"}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return jsonify({
                "final_response": data.get("response", ""),
                "sources": data.get("source_phones", []),
                "stage_1_technical_data": data.get("metadata", {}).get("stage_1_analyst"),
                "stage_2_review": data.get("metadata", {}).get("stage_2_reviewer")
            })
    except Exception:
        pass

    # 2. Local fallback if FastAPI server is starting or offline
    try:
        from sequential_workflow import run_sequential_workflow
        result = run_sequential_workflow(q)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Error running multi-agent workflow: {str(exc)}"}), 500


@app.route("/api/phones", methods=["GET"])
def get_phones():
    """Return all scraped Samsung phones."""
    data = load_data()
    return jsonify(data)


@app.route("/api/raw-json", methods=["GET"])
def get_raw_json():
    """Download or view the raw scraped JSON file."""
    if os.path.exists(DATA_PATH):
        return send_file(DATA_PATH, mimetype="application/json", as_attachment=False)
    return jsonify({"error": "Dataset file not found"}), 404


@app.route("/api/query", methods=["GET"])
def query_phones():
    """Query and filter phones based on query parameters."""
    data = load_data()
    phones = data.get("phones", [])

    q = request.args.get("q", "").strip().lower()
    series = request.args.get("series", "").strip()
    min_battery = request.args.get("min_battery", type=int)
    chipset = request.args.get("chipset", "").strip().lower()
    camera_mp = request.args.get("camera_mp", type=int)

    results = []
    for phone in phones:
        kf = phone.get("key_features", {})

        # Series filter
        if series and series.lower() not in phone.get("series", "").lower():
            continue

        # Min battery filter
        if min_battery:
            cap_num = kf.get("battery_life", {}).get("capacity_numeric_mah")
            if not cap_num or cap_num < min_battery:
                continue

        # Chipset filter
        if chipset:
            p_chipset = kf.get("processor", {}).get("chipset", "").lower()
            if chipset not in p_chipset:
                continue

        # Camera megapixel filter
        if camera_mp:
            p_cam = kf.get("camera_specs", {}).get("main_camera", {}).get("primary_sensor", "")
            if f"{camera_mp} mp" not in p_cam.lower():
                continue

        # Free text search
        if q:
            searchable_text = " ".join([
                phone.get("model_name", ""),
                phone.get("series", ""),
                kf.get("display_size", ""),
                kf.get("processor", {}).get("chipset", ""),
                kf.get("processor", {}).get("cpu", ""),
                kf.get("camera_specs", {}).get("main_camera", {}).get("setup", ""),
                kf.get("camera_specs", {}).get("main_camera", {}).get("primary_sensor", ""),
                kf.get("battery_life", {}).get("capacity", "")
            ]).lower()
            if q not in searchable_text:
                continue

        results.append(phone)

    return jsonify({
        "query": {
            "q": q,
            "series": series,
            "min_battery": min_battery,
            "chipset": chipset,
            "camera_mp": camera_mp
        },
        "count": len(results),
        "results": results
    })


if __name__ == "__main__":
    print("=" * 60)
    print(" Samsung Phone Query and Review System UI")
    print(" Server running at: http://localhost:4005")
    print("=" * 60)
    app.run(host="0.0.0.0", port=4005, debug=False)
