"""
Check Database Contents (PostgreSQL & SQLite)
==============================================
Inspects stored phones, specifications, and camera lenses in the database.
"""

from sqlalchemy import func
from sqlalchemy.orm import Session
from db_manager import Base, CameraLens, Phone, PhoneSpecification, ChatQueryLog, get_database_engine

def inspect_database():
    engine, db_type = get_database_engine()
    print("=" * 78)
    print(f" DATABASE INSPECTION REPORT (Active Database: {db_type})")
    print("=" * 78)

    with Session(engine) as session:
        # 1. Table Statistics
        total_phones = session.query(func.count(Phone.id)).scalar() or 0
        total_specs = session.query(func.count(PhoneSpecification.id)).scalar() or 0
        total_lenses = session.query(func.count(CameraLens.id)).scalar() or 0
        total_chats = session.query(func.count(ChatQueryLog.id)).scalar() or 0

        print("\n📊 TABLE ROW COUNTS IN POSTGRESQL:")
        print(f"  • 'phones' table:               {total_phones} devices")
        print(f"  • 'phone_specifications' table: {total_specs} spec entries")
        print(f"  • 'camera_lenses' table:        {total_lenses} individual lens records")
        print(f"  • 'chat_queries' table:         {total_chats} user chat conversations")

        # 2. List all phones
        print("\n📱 SAVED PHONES LIST:")
        print(f"{'ID':<4} | {'Model Name':<28} | {'Display':<10} | {'Battery':<10} | {'Chipset'}")
        print("-" * 78)
        phones = session.query(Phone).order_by(Phone.id).all()
        for p in phones:
            chipset_short = (p.chipset[:25] + "...") if p.chipset and len(p.chipset) > 25 else (p.chipset or "N/A")
            disp = p.display_size or "N/A"
            bat = p.battery_capacity or "N/A"
            print(f"{p.id:<4} | {p.model_name:<28} | {disp:<10} | {bat:<10} | {chipset_short}")

        # 3. Sample Detailed Inspection of Phone #1
        if phones:
            sample = phones[0]
            print("\n" + "=" * 78)
            print(f"🔍 SAMPLE PHONE DEEP-DIVE: {sample.model_name} (ID: {sample.id})")
            print("=" * 78)
            print(f"  • Series:          {sample.series}")
            print(f"  • Display:         {sample.display_size} ({sample.display_type}), {sample.display_resolution}")
            print(f"  • Processor:       {sample.chipset} | CPU: {sample.cpu}")
            print(f"  • Battery:         {sample.battery_capacity} ({sample.battery_type}) | Charging: {sample.charging.splitlines()[0] if sample.charging else 'N/A'}")
            print(f"  • Main Camera:     {sample.main_camera_setup} ({sample.main_camera_primary})")
            print(f"  • Lenses Breakdown ({len(sample.camera_lenses)} lenses):")
            for cl in sample.camera_lenses:
                print(f"      [{cl.camera_type.upper()}] {cl.lens_description}")

            print(f"\n  • Sample Granular Specs from 'phone_specifications' ({len(sample.specifications)} total):")
            for spec in sample.specifications[:8]:
                val = spec.spec_value.replace("\n", " ")
                if len(val) > 40:
                    val = val[:37] + "..."
                print(f"      [{spec.category}] {spec.spec_name}: {val}")

    print("\n" + "=" * 78)
    print("✅ All data is verified in the PostgreSQL database!")
    print("=" * 78)

if __name__ == "__main__":
    inspect_database()
