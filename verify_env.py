
import sys
import pkg_resources

required = {'geopandas', 'pandas', 'numpy', 'networkx', 'matplotlib', 'openpyxl', 'seaborn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Missing packages: {missing}")
    sys.exit(1)
else:
    print("All required packages installed successfully.")
    sys.exit(0)
