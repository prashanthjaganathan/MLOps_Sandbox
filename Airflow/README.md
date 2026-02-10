Understanding the Satellite Data
Before we generate data, let me explain what we're simulating.
What is Satellite Telemetry?
Telemetry = Real-time data sent from a satellite to ground control, telling us:
Is the satellite healthy?
Are systems working normally?
Are there any problems?
Think of it like a health monitoring app on your phone, but for a satellite in space! 

Why These Metrics?
Real satellites track hundreds of metrics, but these 5 are critical:
Temperature - Too hot → Electronics fail. Too cold → Batteries freeze.
Voltage - Low voltage → Systems shut down
Current - Sudden spike → Electrical short
Angular Velocity - Spinning too fast → Lost control
Solar Power - No power → Dead satellite
What Are "Anomalies"?
An anomaly is an unusual reading that might indicate a problem.
Examples:
# Normal operationtemperature = 20.0°C   ✅voltage = 28.0V        ✅current = 5.0A         ✅# ANOMALY! Something is wrong!temperature = 42.0°C   🚨 Overheating!voltage = 22.0V        🚨 Battery dying!current = 8.5A         🚨 Short circuit!
Our ML model will learn to detect these anomalies automatically! 🤖


 Understanding the Complete Pipeline
Pipeline Architecture:
📡 Ingest → 🧹 Preprocess → ⚙️ Feature Engineering → 🤖 Train Model → 🎯 Score Data → 📝 Report
Detailed breakdown:
Step	Task	What It Does	Input	Output
1	Ingest	Load raw telemetry	CSV path	CSV path
2	Preprocess	Clean & resample to hourly	Raw CSV	Cleaned CSV
3	Feature Engineering	Create rolling stats, deltas	Cleaned CSV	Features CSV
4	Train Model	Train Isolation Forest	Features CSV	Model (.pkl)
5	Score Data	Predict anomalies	Features + Model	Scored CSV
6	Generate Report	Create health summary	Scored CSV	Report (.txt)
Step 5.2: Understanding Isolation Forest
Before we code, let's understand what Isolation Forest is and why we use it.
What is Isolation Forest?
The Concept:
Anomalies are "easy to isolate" (they're far from normal data)
Normal data points are "hard to isolate" (they're clustered together)
Analogy:
Imagine you have 100 people in a room:
95 people are standing close together (normal)
5 people are standing alone in corners (anomalies)
Question: If you randomly draw lines to separate people, who gets isolated first?
Answer: The people in corners (anomalies)! They're easier to separate.
Visual Representation:
Normal Data (clustered):        Anomalies (isolated):        🔵🔵🔵🔵                         🔴    🔵🔵🔵🔵                                  🔵🔵🔵🔵                    🔴    🔵🔵🔵🔵                                                                 🔴        Hard to isolate              Easy to isolate    (many neighbors)             (no neighbors)

