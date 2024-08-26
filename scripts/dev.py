def run():
    import subprocess
    
    subprocess.run("poetry run python -m src.main", shell=True)