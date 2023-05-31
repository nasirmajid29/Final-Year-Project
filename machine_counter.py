import subprocess
from collections import defaultdict

# The list of your machines
machines = ["gpu{:02d}".format(i) for i in range(1, 31)]

# Dictionary to hold the counts of logins per user
user_counts = defaultdict(int)

# Iterate over the machines
for machine in machines:
    if machine in ['gpu01','gpu03', 'gpu04', 'gpu24']:
        continue
    # Form the ssh command for nvidia-smi
    cmd_nvidia_smi = f"ssh {machine} nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,nounits,noheader"
    # Form the ssh command for who
    cmd_who = f"ssh {machine} who"

    # Execute the command and get the output
    try:
        total_memory, used_memory, free_memory = subprocess.check_output(cmd_nvidia_smi, shell=True).decode('utf-8').strip().split(", ")
        logged_users = subprocess.check_output(cmd_who, shell=True).decode('utf-8').strip().split("\n")
        logged_users = [user.split()[0] for user in logged_users]
        logged_users = list(set(logged_users))  # Remove duplicates
        print(f"{machine}: {free_memory} MiB free, {used_memory} MiB used, {total_memory} MiB total. Users logged in: {', '.join(logged_users)}")
        for user in logged_users:
            user_counts[user] += 1
    except Exception as e:
        print(f"Failed to get VRAM and logged in users info from {machine}: {str(e)}")

# Print the count of logins per user
for user, count in user_counts.items():
    print(f"{user} is logged in on {count} machines.")