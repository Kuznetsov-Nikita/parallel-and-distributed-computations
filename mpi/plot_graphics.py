import matplotlib.pyplot as plt
import subprocess
import time

processes_count = range(1, 9)
N = [1000, 1000000, 100000000]

plt.figure(figsize=(20, 9))

plt.title("Ускорение от количества процессов", fontsize=30)
plt.xlabel("Количество процессов p", fontsize=20)
plt.ylabel("Ускорение S", fontsize=20)

for n in N:
    s = list()

    for p in processes_count:
        run_command = ['sbatch', '-n', str(p), './run_sbatch_config.sh', str(n)]
        subprocess.run(run_command)

        time.sleep(10)

        file_n, file_p = 0, 0

        while file_n != n or file_p != p:
            file = open('out.txt', 'r')
            data = file.readlines()
            file_n, file_p = int(data[2]), int(data[3])
            file.close()
            time.sleep(5)
            
        s.append(float(data[0]) / float(data[1]))

    plt.plot(processes_count, s)

plt.legend(["N = 1000", "N = 10^6", "N = 10^8"], fontsize=15)
plt.savefig("graphic.png")
