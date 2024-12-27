import subprocess
from multiprocessing import Process
import time
import os

if os.path.exists("./charge1/perm.txt"):
    os.rename("./charge1/perm.txt", "./charge1/perm_old.txt")

if not os.path.exists("./pipe_configs"):
    os.mkdir("./pipe_configs")

def run_config(charge, perm_path="", sleep=None, samp=200, bgd_sz=100):
    print(f"start process {charge}")
    if sleep:
        time.sleep(sleep)
    config = f"""
shap_calculator:
  val_inps_path: val_inps.csv
  model_path: 
  model_type: charge
  mode: charge{charge}
  perm_path: {perm_path} 
  samp: {samp}
  bgd_sz: {bgd_sz}
shap_visualization:
  sv_path: data/y7+1/output.parquet.gzip
  clustering:
    from_which_end: left
    number_of_aa: 4
  filter_expr: 
    """
    with open(f"./pipe_configs/chargepipe_config{charge}.yaml", "w") as file:
        file.write(config)
    subprocess.run(f"python src/shap_prosit/shap_calculator.py ./pipe_configs/chargepipe_config{charge}.yaml",
                   shell=True)

processes = []

p1 = Process(target=run_config, args=(1,))
p1.start()
processes.append(p1)

while not os.path.exists("./charge1/perm.txt"):
    print("Permutation file not created yet, waiting...")
    time.sleep(5)

for num in range(2, 7):
    p = Process(target=run_config, args=(num, "./charge1/perm.txt"))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

os.rename("./charge1/perm.txt", "./charge1/perm_old.txt")