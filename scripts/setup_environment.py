"""
Environment Setup Script
Automated setup for GenAI Autonomous Data Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_gpu_availability():
    """Check if NVIDIA GPU is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def setup_rapids():
    """Setup NVIDIA RAPIDS if GPU is available"""
    if check_gpu_availability():
        print("🚀 NVIDIA GPU detected, installing RAPIDS...")
        
        rapids_packages = [
            "cudf-cu12",
            "cuml-cu12"
        ]
        
        for package in rapids_packages:
            cmd = f"{package} --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple"
            if install_package(cmd):
                print(f"✅ Installed {package}")
            else:
                print(f"❌ Failed to install {package}")
        
        return True
    else:
        print("⚠️ No NVIDIA GPU detected, skipping RAPIDS installation")
        return False

def main():
    """Main setup function"""
    print("🛠️ GenAI Autonomous Data Agent - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install core requirements
    print("\n📦 Installing core requirements...")
    if install_package("-r requirements.txt"):
        print("✅ Core requirements installed")
    else:
        print("❌ Failed to install core requirements")
        sys.exit(1)
    
    # Setup RAPIDS (optional)
    print("\n🚀 Checking GPU availability...")
    gpu_available = setup_rapids()
    
    # Create directories
    print("\n📁 Creating directory structure...")
    directories = [
        "data/raw", "data/processed", "data/sample_datasets",
        "models/trained_models", "models/explainers", "models/artifacts",
        "outputs/reports", "outputs/visualizations", "outputs/exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Generate sample data
    print("\n📊 Generating sample datasets...")
    try:
        from data.sample_datasets.sample_generator import create_all_sample_datasets
        create_all_sample_datasets()
        print("✅ Sample datasets created")
    except Exception as e:
        print(f"❌ Error creating sample datasets: {e}")
    
    # Setup complete
    print("\n🎉 Environment setup completed!")
    print(f"GPU Acceleration: {'✅ ENABLED' if gpu_available else '❌ DISABLED'}")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run: streamlit run app.py")
    print("3. Open browser to: http://localhost:8501")

if __name__ == "__main__":
    main()
