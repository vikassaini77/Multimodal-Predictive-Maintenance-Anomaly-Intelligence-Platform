import time
import random
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_demo_scenario():
    logger.info("Initializing 30-minute factory simulation scenario...")
    
    # Simulate normal operation
    logger.info("Phase 1: Normal Operation (0-15 mins)")
    for i in range(3):
        logger.info(f"T+{i*5} mins: Vibration sensors nominal (0.2g RMS). Cameras clear.")
        time.sleep(1)
        
    # Simulate sensor drift
    logger.info("Phase 2: Sensor Drift Detected (15-25 mins)")
    logger.warning("T+15 mins: Conveyor C-4 bearing vibration increasing (0.5g RMS).")
    time.sleep(1)
    logger.warning("T+20 mins: Conveyor C-4 bearing vibration critical (1.2g RMS). Acoustic anomalies detected.")
    time.sleep(1)
    logger.warning("T+25 mins: Thermal camera detects bearing housing at 85°C (Threshold: 70°C).")
    time.sleep(1)
    
    # Trigger Anomaly & GraphSAGE prediction
    logger.info("Phase 3: Anomaly Tripped & Fault Propagation (25-28 mins)")
    logger.error("T+26 mins: MULTIMODAL FUSION ALERT - Conveyor C-4 Bearing Failure Imminent.")
    time.sleep(1)
    logger.error("T+27 mins: GraphSAGE prediction: 85% probability of cascade failure to Assembly Arm A-2 within 4 hours.")
    time.sleep(1)
    
    # Agent Diagnosis
    logger.info("Phase 4: Agentic Diagnosis & RAG (28-30 mins)")
    logger.info("T+28 mins: Autonomous ReAct Agent triggered.")
    time.sleep(1)
    logger.info("Agent: [Action] query_sensor_history(machine_id='C-4')")
    logger.info("Agent: [Observation] Vibration spiked 600% in last 10 mins.")
    time.sleep(1)
    logger.info("Agent: [Action] retrieve_manual(query='C-4 bearing replacement overheating')")
    logger.info("Agent: [Observation] OEM manual suggests replacing SKF-6205 bearing and checking lubrication lines.")
    time.sleep(1)
    
    # Final Output
    logger.info("Agent: [Final Diagnosis] Bearing SKF-6205 on Conveyor C-4 is failing due to suspected lubrication loss. Immediate replacement required to prevent downstream shutdown of Assembly Arm A-2. Part #SKF-6205 is in stock (Bin 4A).")
    logger.info("Scenario complete. Data populated for UI demo.")

if __name__ == "__main__":
    generate_demo_scenario()
