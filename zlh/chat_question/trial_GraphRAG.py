import os
import subprocess

os.chdir('/home/student/zlh/GuardBot-main/Graph_RAG/ragtest')
print()

# 定义要执行的命令
commands = [
    "graphrag query --root ./ --method drift --query '保险法需要学什么?'"
]

# 遍历命令并执行
for command in commands:
    try:
        # 执行命令
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 打印输出结果
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        # 打印错误信息
        print("An error occurred while executing the command:", e)
        print("Error output:", e.stderr)