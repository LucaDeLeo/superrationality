#!/usr/bin/env python3
"""
Rate limit handler with checkpoint functionality
"""
import time
import json
import os
from datetime import datetime, timedelta

class RateLimitHandler:
    def __init__(self, checkpoint_file=".rate_limit_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self.load_checkpoint()
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"last_request": None, "reset_time": None, "completed_tasks": []}
    
    def save_checkpoint(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def wait_for_reset(self, hours=2):
        reset_time = datetime.now() + timedelta(hours=hours)
        self.checkpoint["reset_time"] = reset_time.isoformat()
        self.save_checkpoint()
        
        print(f"Rate limit reached. Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Option 1: Use at command
        os.system(f'echo "cd {os.getcwd()} && python {__file__} --resume" | at {reset_time.strftime("%H:%M %m/%d/%Y")}')
        
        # Option 2: Create a cron job
        # self.create_cron_job(reset_time)
        
        print("Task scheduled. You can safely exit.")
    
    def resume_work(self):
        print("Resuming work after rate limit reset...")
        print(f"Completed tasks: {self.checkpoint['completed_tasks']}")
        # Continue your work here

if __name__ == "__main__":
    import sys
    handler = RateLimitHandler()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        handler.resume_work()
    else:
        handler.wait_for_reset(2)