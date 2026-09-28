import sys

def check_python_environment():
    # If base_prefix and prefix are different, you are in a virtual environment
    is_venv = sys.prefix != sys.base_prefix
    
    print("=" * 50)
    if is_venv:
        print("🟢 SUCCESS: You are running inside a VIRTUAL ENVIRONMENT!")
        print(f"Active Environment Path: {sys.prefix}")
    else:
        print("🔴 WARNING: You are running on your GLOBAL/LOCAL machine!")
        print(f"Global Python Path: {sys.base_prefix}")
    print("=" * 50)

if __name__ == "__main__":
    check_python_environment()
