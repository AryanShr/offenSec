import os
import subprocess

def is_dotnet_assembly(file_path):
    try:
        output = subprocess.check_output(f'ildasm.exe /nobar "{file_path}"', stderr=subprocess.STDOUT, shell=True)
        return '.assembly' in output.decode('utf-8')
    except subprocess.CalledProcessError:
        return False

def count_dotnet_assemblies(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.exe'):
                file_path = os.path.join(root, file)
                if is_dotnet_assembly(file_path):
                    count += 1
    return count

directory = 'TestData\\malware'  # replace with your directory
print(f'The number of .NET executables in the directory is: {count_dotnet_assemblies(directory)}')
