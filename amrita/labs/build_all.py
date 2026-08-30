import os
import sys

scripts = [
    "generate_lab1.py",
    "generate_lab2.py",
    "generate_lab3.py",
    "generate_lab4.py"
]

print("Starting generation of all 4 lab notebooks...")
for script in scripts:
    print(f"\n---> Running {script}...")
    full_path = os.path.join(r"C:\github\drl\amrita\labs", script)
    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {'__name__': '__main__', '__file__': full_path})
    print(f"---> Finished {script}")

print("\nALL 4 LAB NOTEBOOKS SUCCESSFULLY GENERATED AND EXECUTED!")
