# CL1-1: Compose a Python program to determine your computer’s machine epsilon.

machine_epsilon = 2**0
count = 0;

while (1 + machine_epsilon != 1):
    machine_epsilon /= 2;
    count+=1;

print(f"Machine Epsilon = 2^({count-1})")
print(f"{machine_epsilon*2:.16f}")
