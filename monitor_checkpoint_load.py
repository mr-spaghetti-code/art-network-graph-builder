#!/usr/bin/env python3
"""Monitor if a Python process is actively loading a checkpoint"""

import psutil
import time
import sys

def monitor_process(process_name="python"):
    """Monitor CPU and memory usage of Python processes"""
    print(f"Monitoring {process_name} processes...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            found_process = False
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        # Get more detailed info
                        proc.cpu_percent(interval=0.1)  # Initialize CPU monitoring
                        time.sleep(0.1)
                        cpu = proc.cpu_percent(interval=0.1)
                        mem = proc.info['memory_info'].rss / (1024**3)  # GB
                        
                        if cpu > 0.1 or mem > 0.5:  # Show only active processes
                            cmdline = ' '.join(proc.cmdline()[:3]) if proc.cmdline() else proc.info['name']
                            print(f"PID {proc.info['pid']}: CPU {cpu:5.1f}% | RAM {mem:5.1f} GB | {cmdline}")
                            found_process = True
                            
                            # Check if it's likely loading JSON
                            if mem > 2.0 and cpu > 50:
                                print("  ⚡ High CPU & Memory - likely parsing JSON")
                            elif mem > 3.0 and cpu < 10:
                                print("  💾 High memory, low CPU - might be stuck or doing I/O")
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if found_process:
                print("-" * 60)
            else:
                print("No active Python processes found...")
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_process() 