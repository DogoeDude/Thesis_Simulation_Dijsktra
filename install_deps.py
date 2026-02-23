import subprocess
import sys
import pathlib

REQ_FILE = pathlib.Path(__file__).with_name('requirements.txt')

def install_requirements():
    if not REQ_FILE.is_file():
        print('requirements.txt not found')
        sys.exit(1)
    cmd = [sys.executable, '-m', 'pip', 'install', '--user', '-r', str(REQ_FILE)]
    subprocess.check_call(cmd)
    print('Dependencies installed globally (user site-packages).')

if __name__ == "__main__":
    install_requirements()
