'''
Application exchanges data with ULTRA96 FPGA board running the model for inference
'''

from ftpretty import ftpretty
import numpy as np
import os
import socket


def place_test_file_in_fpga(file_path, host_addr='192.168.43.3', remote_file_path='/home/root/Vitis-AI/dex_read'):
    """
    File copies the file pointed to by file_path to the remote_file_path on the server pointed to by host_addr
    """

    # Supply the credentisals
    f = ftpretty(host_addr, "root", "root")

    file = np.load(file_path)  # read in the numpy file
    
    # Get a file, save it locally
    # f.get('someremote/file/on/server.txt', '/tmp/localcopy/server.txt')

    # Put a local file to a remote location
    # non-existent subdirectories will be created automatically
    f.put(file_path, remote_file_path+'/')


def launch_sensor_server():
    """
    Function starts a TCP server and waits for connection from the inference FPGA client

    Mode of operation:
    client(ULTRA96) ---> datax<N> ----> server (computer)
    server : writes  <N> files to the data exchange read directory on the ULTRA96
    server ---> datax<N>c ----> client
    client : inference and writes result to data exchange directory write directory on the ULTRA96, deletes files in read directory
    client ----> inference<N>c ---> server
    server : reads in inference data from data exchange write directory of the ULTRA96, and displays visual output
    """

    HOST = '192.168.43.3'  # ULTRA96
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)

if __name__ == "__main__":

    f_path_root = '/home/sambit/data/KITTI/validation/velodyne_2d'
    files = os.listdir(f_path_root)

    # copy 100 files over
    for i in range(50):
        place_test_file_in_fpga(f_path_root + '/' + files[i])
