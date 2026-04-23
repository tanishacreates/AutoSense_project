# AutoSense —  Driver Assistance System
Hey there. 
Accidents are increasing day by day. Drive Assistance


## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run integration tests (no camera needed)
python tests/test_integration.py

# 4. Run the full system
python main.py

# No camera? Headless mode?
python main.py --no-display --no-dashboard

# Change sensitivity
python main.py --sensitivity high

# Simulate drowsy driving sensor data
python main.py --sensor-pattern drowsy

# Save Person 1's original event CSV
python main.py --save-log

# Evaluate a saved session
python main.py --evaluate logs/session_YYYYMMDD_HHMMSS.csv
