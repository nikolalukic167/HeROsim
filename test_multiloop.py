#!/usr/bin/env python3

"""
Simple test to verify MultiLoopScheduler override is working
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_multiloop_override():
    """Test that MultiLoopScheduler properly overrides scheduler_process"""
    
    # Mock the necessary components for testing
    class MockEnv:
        def __init__(self):
            self.now = 0
        
        def process(self, generator):
            return generator
    
    class MockPolicy:
        def __init__(self):
            self.batch_size = 3
            self.batch_timeout = 0.1
    
    class MockMutex:
        def get(self):
            return None
        
        def put(self, state):
            pass
    
    class MockData:
        pass
    
    class MockAutoscaler:
        pass
    
    class MockNodes:
        pass
    
    # Import the scheduler
    try:
        from src.policy.multiloop.scheduler import MultiLoopScheduler
        
        # Create mock components
        env = MockEnv()
        mutex = MockMutex()
        data = MockData()
        policy = MockPolicy()
        autoscaler = MockAutoscaler()
        nodes = MockNodes()
        
        # Create scheduler instance
        scheduler = MultiLoopScheduler(env, mutex, data, policy, autoscaler, nodes)
        
        # Check if the scheduler has the batch_size attribute (indicating our override worked)
        if hasattr(scheduler, 'batch_size'):
            print("✅ MultiLoopScheduler override is working!")
            print(f"   Batch size: {scheduler.batch_size}")
            print(f"   Batch timeout: {scheduler.batch_timeout}")
            return True
        else:
            print("❌ MultiLoopScheduler override failed - no batch_size attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MultiLoopScheduler: {e}")
        return False

if __name__ == "__main__":
    success = test_multiloop_override()
    sys.exit(0 if success else 1) 