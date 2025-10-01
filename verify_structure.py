"""
Project Structure Verification Script
"""

from pathlib import Path

def verify_project_structure():
    """Verify all required files and directories exist"""
    
    required_structure = {
        # Root files
        'app.py': 'file',
        'setup.py': 'file', 
        'requirements.txt': 'file',
        'config.yaml': 'file',
        '.env.template': 'file',
        'README.md': 'file',
        
        # Source directory
        'src/': 'dir',
        'src/__init__.py': 'file',
        'src/config.py': 'file',
        'src/data_processor.py': 'file',
        'src/eda_module.py': 'file',
        'src/ml_module.py': 'file',
        'src/explain_module.py': 'file',
        'src/sql_agent.py': 'file',
        'src/report_generator.py': 'file',
        'src/utils.py': 'file',
        
        # Data directories
        'data/': 'dir',
        'data/raw/': 'dir',
        'data/processed/': 'dir',
        'data/sample_datasets/': 'dir',
        'data/sample_datasets/sample_generator.py': 'file',
        
        # Model directories  
        'models/': 'dir',
        'models/trained_models/': 'dir',
        'models/explainers/': 'dir',
        'models/artifacts/': 'dir',
        
        # Notebooks
        'notebooks/': 'dir',
        'notebooks/01_colab_prototype.ipynb': 'file',
        'notebooks/02_eda_analysis.ipynb': 'file',
        
        # Tests
        'tests/': 'dir',
        'tests/__init__.py': 'file',
        'tests/test_data_processor.py': 'file',
        'tests/test_ml_module.py': 'file',
        
        # Documentation
        'docs/': 'dir',
        'docs/api_documentation.md': 'file',
        'docs/user_manual.md': 'file',
        
        # Scripts
        'scripts/': 'dir',
        'scripts/setup_environment.py': 'file',
        'scripts/download_sample_data.py': 'file',
        
        # Templates
        'templates/': 'dir',
        'templates/report_template.html': 'file',
        'templates/dashboard_theme.json': 'file',
        
        # Outputs and logs
        'outputs/': 'dir',
        'outputs/reports/': 'dir',
        'outputs/visualizations/': 'dir',
        'outputs/exports/': 'dir',
        'logs/': 'dir'
    }
    
    print("🔍 Verifying project structure...")
    print("=" * 50)
    
    missing_items = []
    
    for item_path, item_type in required_structure.items():
        path_obj = Path(item_path)
        
        if item_type == 'file':
            if path_obj.is_file():
                print(f"✅ {item_path}")
            else:
                print(f"❌ {item_path} (MISSING FILE)")
                missing_items.append(item_path)
                
        elif item_type == 'dir':
            if path_obj.is_dir():
                print(f"✅ {item_path}")
            else:
                print(f"❌ {item_path} (MISSING DIRECTORY)")
                missing_items.append(item_path)
    
    print("\n" + "=" * 50)
    
    if missing_items:
        print(f"❌ {len(missing_items)} items missing:")
        for item in missing_items:
            print(f"   • {item}")
        print("\n💡 Run the setup script to fix missing items:")
        print("   python setup.py")
    else:
        print("🎉 All required files and directories are present!")
        print("✅ Project structure is complete!")
        
        # Check for sample data
        sample_files = [
            'data/sample_datasets/ecommerce_sales.csv',
            'data/sample_datasets/healthcare_data.csv', 
            'data/sample_datasets/financial_market.csv',
            'data/sample_datasets/automotive_performance.csv'
        ]
        
        sample_count = sum(1 for f in sample_files if Path(f).exists())
        print(f"\n📊 Sample datasets: {sample_count}/4 available")
        
        if sample_count == 0:
            print("💡 Generate sample datasets:")
            print("   cd data/sample_datasets && python sample_generator.py")
    
    return len(missing_items) == 0

if __name__ == "__main__":
    verify_project_structure()
