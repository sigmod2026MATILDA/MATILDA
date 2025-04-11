# MATILDA

## Installation

### Download database requirements
#### macOS
```bash
brew install mysql-client@8.4
```

#### Linux
```bash
sudo apt-get install mysql-client-8.4
```

#### Windows
```powershell
choco install mysql --version=8.4
```

### Set up Python3 virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip3 install -r requirements.txt
```

### Run the application
1. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
2. Start the application:
```bash
cd src 
python main.py
```



