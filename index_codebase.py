import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INDEX_FILE = os.path.join(PROJECT_ROOT, "code_index.txt")
print(f"Writing index to: {INDEX_FILE}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"INDEX_FILE: {INDEX_FILE}")

def index_py_files():
    py_files = []
    for root, _, files in os.walk(PROJECT_ROOT):
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.relpath(os.path.join(root, f), PROJECT_ROOT))
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as out:
            for path in sorted(py_files):
                out.write(path + "\n")
        print(f"Indexed {len(py_files)} Python files to {INDEX_FILE}")
    except Exception as e:
        import traceback
        print("Error writing index file:", e)
        traceback.print_exc()
    print(f"File exists: {os.path.isfile(INDEX_FILE)}")
    print(f"Absolute path: {os.path.abspath(INDEX_FILE)}")

if __name__ == "__main__":
    index_py_files()
