import random

def generate_synthetic_manuals():
    docs = []
    
    # Document 1: Conveyor Belt
    docs.append({
        "title": "Conveyor Belt Operation and Maintenance",
        "content": (
            "## Section 1: Overview\n"
            "The conveyor belt system is designed for continuous material transport. "
            "It operates via a dual-motor drive.\n"
            "## Section 2: Fault Diagnosis - Misalignment\n"
            "Belt misalignment can be detected through increased motor temperature and unusual vibration patterns. "
            "If misalignment occurs, inspect the idler rollers and adjust the tensioning system. "
            "Failure to correct misalignment may lead to catastrophic belt tear.\n"
            "## Section 3: Sensor Anomalies\n"
            "Temperature sensors exceeding 85C for more than 5 minutes typically indicate bearing wear or friction issues."
        )
    })
    
    # Document 2: Hydraulic Pump
    docs.append({
        "title": "Hydraulic Pump Incident Report #4421",
        "content": (
            "## Incident Summary\n"
            "On Oct 12th, the primary hydraulic pump failed, causing a 4-hour line stoppage. "
            "Prior to failure, acoustic sensors detected high-frequency whining (above 15kHz). "
            "## Root Cause Analysis\n"
            "The root cause was identified as fluid cavitation due to a clogged intake filter. "
            "## Corrective Actions\n"
            "Filters must be inspected every 200 hours. Acoustic anomalies in the 15kHz range should immediately trigger a shutdown."
        )
    })
    
    # Document 3: General Motor Maintenance
    docs.append({
        "title": "AC Motor Servicing Guidelines",
        "content": (
            "## Routine Inspection\n"
            "AC motors should undergo visual and thermal inspection weekly. "
            "Ensure ventilation grilles are free of debris.\n"
            "## Bearing Replacement\n"
            "Bearings should be replaced every 10,000 operational hours or if vibration exceeds 2.5 mm/s RMS. "
            "Use only OEM-specified synthetic grease."
        )
    })
    
    return docs
