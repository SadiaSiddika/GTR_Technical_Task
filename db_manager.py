"""
Samsung Phone Database Manager & Ingestion Pipeline
===================================================
Uses SQLAlchemy to store scraped Samsung Galaxy phone data into a relational database.

Dynamic Connection Strategy:
- Primary: PostgreSQL (production connection via DATABASE_URL or POSTGRES_* env vars)
- Fallback: Embedded SQLite database ('samsung_phones.db') for local offline development

Tables:
- phones: Core device metadata & indexed key features (Display size, Camera, Battery, Processor)
- phone_specifications: Granular category-wise specs (Display, Platform, Memory, Body, Sensors, etc.)
- camera_lenses: Individual camera lens breakdowns
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DatabaseManager")


# ==============================================================================
# 1. SQLAlchemy ORM Models
# ==============================================================================

class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy ORM."""
    pass


class Phone(Base):
    """Represents a Samsung Galaxy smartphone with primary key features."""
    __tablename__ = "phones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    series: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    brand: Mapped[str] = mapped_column(String(50), default="Samsung", nullable=False)
    url: Mapped[str] = mapped_column(String(255), nullable=False)
    image_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    scraped_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Key Features: Display
    display_size: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    display_size_numeric: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    display_type: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    display_resolution: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    display_protection: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    # Key Features: Processor
    chipset: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    cpu: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gpu: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    operating_system: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Key Features: Camera
    main_camera_setup: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    main_camera_primary: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    main_camera_video: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    selfie_camera_setup: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    selfie_camera_primary: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    selfie_camera_video: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Key Features: Battery Life
    battery_capacity: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    battery_capacity_mah: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    battery_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    charging: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active_use_score: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    endurance_rating: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Hardware & Build
    dimensions: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    weight: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_rating: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    internal_storage_and_ram: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    announced: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relational associations
    specifications: Mapped[List["PhoneSpecification"]] = relationship(
        "PhoneSpecification",
        back_populates="phone",
        cascade="all, delete-orphan",
        order_by="PhoneSpecification.id"
    )
    camera_lenses: Mapped[List["CameraLens"]] = relationship(
        "CameraLens",
        back_populates="phone",
        cascade="all, delete-orphan",
        order_by="CameraLens.id"
    )

    def __repr__(self) -> str:
        return f"<Phone(id={self.id}, model='{self.model_name}', chipset='{self.chipset}', battery='{self.battery_capacity}')>"


class PhoneSpecification(Base):
    """Detailed category-wise specifications (Display, Platform, Body, Sensors, etc.)."""
    __tablename__ = "phone_specifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("phones.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    spec_name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    spec_value: Mapped[str] = mapped_column(Text, nullable=False)

    phone: Mapped["Phone"] = relationship("Phone", back_populates="specifications")

    def __repr__(self) -> str:
        return f"<Spec(phone_id={self.phone_id}, category='{self.category}', key='{self.spec_name}')>"


class CameraLens(Base):
    """Individual camera lens specs for main and selfie modules."""
    __tablename__ = "camera_lenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("phones.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'main' or 'selfie'
    lens_description: Mapped[str] = mapped_column(Text, nullable=False)

    phone: Mapped["Phone"] = relationship("Phone", back_populates="camera_lenses")


class ChatQueryLog(Base):
    """Stores user queries, selected engine route, and generated answers."""
    __tablename__ = "chat_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str] = mapped_column(String(50), nullable=False)  # 'simple_rag' or 'crewai_multi_agent'
    response: Mapped[str] = mapped_column(Text, nullable=False)
    source_phones: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    execution_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ChatQueryLog(id={self.id}, question='{self.question[:30]}...', route='{self.route}')>"


def log_chat_query(
    question: str,
    route: str,
    response: str,
    source_phones: Optional[List[str]] = None,
    execution_time: Optional[float] = None
) -> None:
    """Saves user question and chatbot answer into the PostgreSQL/SQLite database."""
    try:
        engine, _ = get_database_engine()
        with Session(engine) as session:
            entry = ChatQueryLog(
                question=question,
                route=route,
                response=response,
                source_phones=", ".join(source_phones) if source_phones else None,
                execution_time_seconds=execution_time,
                created_at=datetime.utcnow()
            )
            session.add(entry)
            session.commit()
    except Exception as exc:
        logger.warning("Could not log chat query to database: %s", exc)


# ==============================================================================
# 2. Dynamic Database Connection Strategy
# ==============================================================================

def get_database_engine(sqlite_db_name: str = "samsung_phones.db") -> Tuple[Engine, str]:
    """
    Connect to PostgreSQL if available; gracefully fall back to SQLite.
    
    Returns:
        (Engine, connection_type_str)
    """
    # 1. Resolve PostgreSQL connection string from environment or defaults
    pg_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if not pg_url:
        pg_user = os.environ.get("POSTGRES_USER", "postgres")
        pg_password = os.environ.get("POSTGRES_PASSWORD", "admin")
        pg_host = os.environ.get("POSTGRES_HOST", "localhost")
        pg_port = os.environ.get("POSTGRES_PORT", "5432")
        pg_db = os.environ.get("POSTGRES_DB", "samsung_phones")
        pg_url = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    elif pg_url.startswith("postgresql://"):
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg://", 1)

    # Try connecting to PostgreSQL
    try:
        logger.info("Attempting primary connection to PostgreSQL: %s ...", pg_url.split("@")[-1])
        pg_engine = create_engine(
            pg_url,
            connect_args={"connect_timeout": 3},
            pool_pre_ping=True
        )
        # Verify connection with a quick ping
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(">>> SUCCESS: Connected to primary PostgreSQL database.")
        return pg_engine, "PostgreSQL"
    except Exception as exc:
        logger.warning(
            "Primary PostgreSQL connection could not be established (%s).",
            str(exc).strip().split("\n")[0]
        )
        logger.info(">>> GRACEFUL FALLBACK: Initializing embedded SQLite database for local offline development...")

    # 2. Fallback to Embedded SQLite
    sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sqlite_db_name)
    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
    with sqlite_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info(">>> SUCCESS: Active database is SQLite: %s", sqlite_path)
    return sqlite_engine, "SQLite"


# ==============================================================================
# 3. Data Ingestion Pipeline
# ==============================================================================

class SamsungDataIngester:
    """Reads scraped JSON and ingests it into the relational database."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionFactory = sessionmaker(bind=engine)

    def init_tables(self) -> None:
        """Create all tables if they do not exist."""
        logger.info("Creating database tables if not present...")
        Base.metadata.create_all(self.engine)
        logger.info("Database tables initialized successfully.")

    def ingest_from_json(self, json_filepath: str) -> int:
        """
        Parse JSON file and insert all phones, specifications, and camera lenses.
        Handles duplicates by updating existing records.
        """
        if not os.path.exists(json_filepath):
            logger.error("JSON data file not found at: %s", json_filepath)
            return 0

        with open(json_filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        phones_data = payload.get("phones", [])
        total = len(phones_data)
        logger.info("Beginning relational ingestion of %d phones from '%s'...", total, json_filepath)

        inserted_count = 0
        with self.SessionFactory() as session:
            for item in phones_data:
                model_name = item.get("model_name")
                if not model_name:
                    continue

                # Parse timestamp
                scraped_dt = None
                if item.get("scraped_at"):
                    try:
                        scraped_dt = datetime.fromisoformat(item["scraped_at"].replace("Z", "+00:00"))
                    except Exception:
                        scraped_dt = datetime.utcnow()

                kf = item.get("key_features", {})
                disp = kf.get("display", {})
                proc = kf.get("processor", {})
                cams = kf.get("camera_specs", {})
                main_cam = cams.get("main_camera", {})
                selfie_cam = cams.get("selfie_camera", {})
                batt = kf.get("battery_life", {})
                body = item.get("body", {})
                launch = item.get("launch", {})

                # Extract numeric display size for range querying
                disp_size_str = kf.get("display_size", "")
                disp_num_match = re.search(r"(\d+(?:\.\d+)?)", disp_size_str)
                disp_numeric = float(disp_num_match.group(1)) if disp_num_match else None

                # Check if phone already exists
                existing_phone = session.execute(
                    select(Phone).where(Phone.model_name == model_name)
                ).scalar_one_or_none()

                if existing_phone:
                    phone_obj = existing_phone
                    # Clear existing relations to refresh them
                    phone_obj.specifications.clear()
                    phone_obj.camera_lenses.clear()
                else:
                    phone_obj = Phone(model_name=model_name)
                    session.add(phone_obj)

                # Set phone attributes
                phone_obj.series = item.get("series", "Galaxy Series")
                phone_obj.brand = item.get("brand", "Samsung")
                phone_obj.url = item.get("url", "")
                phone_obj.image_url = item.get("image_url")
                phone_obj.scraped_at = scraped_dt

                # Display
                phone_obj.display_size = disp_size_str
                phone_obj.display_size_numeric = disp_numeric
                phone_obj.display_type = disp.get("display_type")
                phone_obj.display_resolution = disp.get("resolution")
                phone_obj.display_protection = disp.get("protection")

                # Processor
                phone_obj.chipset = proc.get("chipset")
                phone_obj.cpu = proc.get("cpu")
                phone_obj.gpu = proc.get("gpu")
                phone_obj.operating_system = proc.get("operating_system")

                # Camera
                phone_obj.main_camera_setup = main_cam.get("setup")
                phone_obj.main_camera_primary = main_cam.get("primary_sensor")
                phone_obj.main_camera_video = main_cam.get("video")
                phone_obj.selfie_camera_setup = selfie_cam.get("setup")
                phone_obj.selfie_camera_primary = selfie_cam.get("primary_sensor")
                phone_obj.selfie_camera_video = selfie_cam.get("video")

                # Battery
                phone_obj.battery_capacity = batt.get("capacity")
                phone_obj.battery_capacity_mah = batt.get("capacity_numeric_mah")
                phone_obj.battery_type = batt.get("battery_type")
                phone_obj.charging = batt.get("charging")
                phone_obj.active_use_score = batt.get("active_use_score")
                phone_obj.endurance_rating = batt.get("endurance_rating")

                # Body & Launch
                phone_obj.dimensions = body.get("dimensions")
                phone_obj.weight = body.get("weight")
                phone_obj.ip_rating = body.get("ip_rating")
                phone_obj.internal_storage_and_ram = item.get("memory", {}).get("internal_storage_and_ram")
                phone_obj.announced = launch.get("announced")
                phone_obj.status = launch.get("status")

                # Ingest Camera Lenses
                for lens_desc in main_cam.get("lenses", []):
                    phone_obj.camera_lenses.append(
                        CameraLens(camera_type="main", lens_description=lens_desc)
                    )
                for lens_desc in selfie_cam.get("lenses", []):
                    phone_obj.camera_lenses.append(
                        CameraLens(camera_type="selfie", lens_description=lens_desc)
                    )

                # Ingest Category-wise detailed specifications
                all_specs = item.get("all_specifications", {})
                for cat_name, cat_dict in all_specs.items():
                    if isinstance(cat_dict, dict):
                        for spec_k, spec_v in cat_dict.items():
                            phone_obj.specifications.append(
                                PhoneSpecification(
                                    category=cat_name,
                                    spec_name=spec_k,
                                    spec_value=str(spec_v)
                                )
                            )

                inserted_count += 1

            session.commit()

        logger.info("Successfully committed %d phones and all relational specs into database.", inserted_count)
        return inserted_count


# ==============================================================================
# 4. Interactive SQL & ORM Query Utilities
# ==============================================================================

class SamsungPhoneQueries:
    """Helper methods demonstrating relational queries for the Query System."""

    def __init__(self, engine: Engine):
        self.SessionFactory = sessionmaker(bind=engine)

    def find_phones_by_processor(self, keyword: str) -> List[Dict[str, Any]]:
        """Filter phones by processor/chipset keyword (e.g., 'Snapdragon', 'Exynos')."""
        with self.SessionFactory() as session:
            stmt = select(Phone).where(Phone.chipset.ilike(f"%{keyword}%")).order_by(Phone.model_name)
            phones = session.scalars(stmt).all()
            return [
                {
                    "model_name": p.model_name,
                    "chipset": p.chipset,
                    "display": p.display_size,
                    "battery": p.battery_capacity
                }
                for p in phones
            ]

    def find_phones_by_battery(self, min_mah: int) -> List[Dict[str, Any]]:
        """Filter phones with battery capacity >= min_mah."""
        with self.SessionFactory() as session:
            stmt = select(Phone).where(Phone.battery_capacity_mah >= min_mah).order_by(Phone.battery_capacity_mah.desc())
            phones = session.scalars(stmt).all()
            return [
                {
                    "model_name": p.model_name,
                    "battery_capacity": p.battery_capacity,
                    "charging": p.charging.split("\n")[0] if p.charging else "N/A"
                }
                for p in phones
            ]

    def find_phones_by_camera_sensor(self, sensor_mp: str) -> List[Dict[str, Any]]:
        """Filter phones by primary camera resolution (e.g. '200 MP', '108 MP')."""
        with self.SessionFactory() as session:
            stmt = select(Phone).where(Phone.main_camera_primary.ilike(f"%{sensor_mp}%")).order_by(Phone.model_name)
            phones = session.scalars(stmt).all()
            return [
                {
                    "model_name": p.model_name,
                    "primary_camera": p.main_camera_primary,
                    "camera_setup": p.main_camera_setup,
                    "chipset": p.chipset.split("(")[0].strip() if p.chipset else "N/A"
                }
                for p in phones
            ]

    def get_full_specs_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete relational details for a phone including lenses and category specs."""
        with self.SessionFactory() as session:
            stmt = select(Phone).where(Phone.model_name == model_name)
            phone = session.scalars(stmt).first()
            if not phone:
                return None

            specs_count = len(phone.specifications)
            lenses = [l.lens_description for l in phone.camera_lenses]

            return {
                "id": phone.id,
                "model_name": phone.model_name,
                "series": phone.series,
                "display": f"{phone.display_size} ({phone.display_type})",
                "processor": f"{phone.chipset} | GPU: {phone.gpu}",
                "main_camera": f"{phone.main_camera_setup} ({phone.main_camera_primary})",
                "camera_lenses_count": len(lenses),
                "battery": f"{phone.battery_capacity} | Charging: {phone.charging.split(chr(10))[0] if phone.charging else 'N/A'}",
                "total_specification_rows": specs_count
            }


# ==============================================================================
# 5. CLI Execution & Verification
# ==============================================================================

def main():
    print("=" * 80)
    print(" Samsung Phone Relational Database Ingestion Pipeline")
    print("=" * 80)

    # 1. Establish connection (PostgreSQL -> SQLite fallback)
    engine, db_type = get_database_engine()

    # 2. Ingest JSON data
    ingester = SamsungDataIngester(engine)
    ingester.init_tables()

    json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samsung_phones_data.json")
    inserted = ingester.ingest_from_json(json_file)

    # 3. Demonstration of Relational Queries
    queries = SamsungPhoneQueries(engine)

    print("\n" + "=" * 80)
    print(f" DEMO QUERIES ON RELATIONAL DATABASE ({db_type})")
    print("=" * 80)

    # Query 1: Snapdragon Flagships
    snapdragons = queries.find_phones_by_processor("Snapdragon")
    print(f"\n1. Phones with Qualcomm Snapdragon Chipset ({len(snapdragons)} found):")
    for p in snapdragons[:4]:
        print(f"   • {p['model_name']} -> {p['chipset'][:40]}... (Battery: {p['battery']})")

    # Query 2: High Capacity Batteries (>= 5000 mAh)
    big_batteries = queries.find_phones_by_battery(5000)
    print(f"\n2. Phones with Battery >= 5000 mAh ({len(big_batteries)} found):")
    for p in big_batteries:
        print(f"   • {p['model_name']} -> {p['battery_capacity']} (Charging: {p['charging']})")

    # Query 3: 200 MP Camera Flagships
    cams_200 = queries.find_phones_by_camera_sensor("200 MP")
    print(f"\n3. Flagship Phones with 200 MP Camera ({len(cams_200)} found):")
    for p in cams_200:
        print(f"   • {p['model_name']} -> {p['primary_camera']} ({p['camera_setup']}) | {p['chipset']}")

    # Query 4: Detailed Model Inspection
    sample_phone = "Samsung Galaxy S24 Ultra"
    details = queries.get_full_specs_summary(sample_phone)
    if details:
        print(f"\n4. Detailed Spec Verification for '{sample_phone}':")
        print(f"   • Display: {details['display']}")
        print(f"   • Processor: {details['processor']}")
        print(f"   • Main Camera: {details['main_camera']} ({details['camera_lenses_count']} lenses stored)")
        print(f"   • Battery: {details['battery']}")
        print(f"   • Relational Specs Rows: {details['total_specification_rows']} detailed attributes")

    print("\n" + "=" * 80)
    print(" Pipeline execution finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
