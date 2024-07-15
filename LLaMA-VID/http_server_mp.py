import os
import json
import argparse
import socket
import multiprocessing as mp
from termcolor import colored
from flask import Flask, request

from importlib.util import find_spec
if find_spec("gpustat") is None: os.system("pip install gpustat")
import GPUtil

def find_free_ports(start_port, num_ports):
    """
    从指定的起始端口开始搜索指定数量的空闲端口。

    :param start_port: 搜索的起始端口
    :param num_ports: 需要找到的空闲端口数量
    :return: 包含空闲端口的列表
    """
    free_ports = []
    current_port = start_port

    while len(free_ports) < num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # 尝试绑定端口
                s.bind(("0.0.0.0", current_port))
                s.listen(1)
                # 如果成功绑定，添加到空闲端口列表
                free_ports.append(current_port)
            except socket.error as e:
                # 如果端口已被占用，则继续尝试下一个端口
                pass
            finally:
                current_port += 1

    return free_ports

def main():
    gpu_list  = [_.id for _ in GPUtil.getGPUs()]
    port_list = find_free_ports(8000, len(gpu_list))

    port_file = 'port.json'
    with open(port_file, 'w') as f:
        json.dump(port_list, f)

    p = mp.Pool(len(gpu_list))
    for gpu_id, port in zip(gpu_list, port_list):
        p.apply_async(
            os.system, 
            args=(f'python http_server.py --gpu-id {gpu_id} --port {port}',),
            error_callback=lambda e: print(colored(e, 'red')),
        )
    p.close()
    p.join()

if __name__ == "__main__":
    main()