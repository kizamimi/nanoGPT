python -m venv venv
venv\\Scripts\\activate.bat
python.exe -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
venv\\Scripts\\deactivate.bat
pause