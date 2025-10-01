"""
Quick import checker for all modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_imports():
    """Check if all modules can be imported"""
    
    modules_to_check = [
        'config',
        'data_processor', 
        'eda_module',
        'ml_module',
        'explain_module',
        'sql_agent',
        'report_generator',
        'utils'
    ]
    
    print("Checking imports...")
    print("=" * 40)
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name} - ERROR: {e}")
        except Exception as e:
            print(f"⚠️ {module_name} - WARNING: {e}")
    
    print("=" * 40)
    print("Import check completed!")

if __name__ == "__main__":
    check_imports()
