How to Integrate into Your Pipeline


############### only for first time dont do it next time or see if you need (important commands are only running it) ###################
Step 1 â€” Replace old file:
bashnano /home/server/fyp_pipeline/session_aggregator.py
Paste entire code above â†’ Ctrl+O â†’ Ctrl+X

Step 2 â€” Delete old checkpoint and CSV to start fresh:
bashrm -f /home/server/fyp_pipeline/last_checkpoint.txt
rm -f /home/server/fyp_pipeline/sessions_output.csv
###########################################################################################################################
Step 3 â€” Test manually first:
bashconda activate myenv
cd /home/server/fyp_pipeline
python3 session_aggregator.py

Step 4 â€” If working, run as service:
bashsudo systemctl restart session-aggregator
sudo systemctl status session-aggregator
sudo journalctl -u session-aggregator -f

Step 5 â€” Verify CSV is being built:
bash# Check row count every minute
watch -n 10 "wc -l /home/server/fyp_pipeline/sessions_output.csv"

Integration Flow
New logs arrive via Filebeat
        â†“
Elasticsearch stores them
        â†“
session-aggregator service wakes up (every 60s)
        â†“
Reads only logs AFTER last checkpoint
        â†“
Builds new sessions
        â†“
Appends to sessions_output.csv
        â†“
Updates checkpoint timestamp
        â†“ (next step)
model_inference.py reads sessions_output.csv
        â†“
Writes predictions to Elasticsearch
        â†“
Kibana dashboard shows alerts
