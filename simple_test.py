#!/usr/bin/env python3

"""
Minimal test to verify method override
"""

# Mock the base Scheduler class
class MockScheduler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def scheduler_process(self):
        print("Base scheduler_process called")
        return "base"

# Create our MultiLoopScheduler
class MultiLoopScheduler(MockScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5
        self.batch_timeout = 0.1
        print("MultiLoopScheduler initialized")
    
    def scheduler_process(self):
        print("MultiLoop scheduler_process called!")
        print(f"Batch size: {self.batch_size}")
        return "multiloop"

# Test the override
def test_override():
    scheduler = MultiLoopScheduler("env", "mutex", "data", "policy", "autoscaler", "nodes")
    
    # Call the method
    result = scheduler.scheduler_process()
    
    print(f"Result: {result}")
    
    if result == "multiloop":
        print("✅ Override is working!")
        return True
    else:
        print("❌ Override failed!")
        return False

if __name__ == "__main__":
    test_override() 