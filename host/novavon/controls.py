import serial
import time
from typing import Union

start_cmd:str = 'START'
stop_cmd:str = 'STOPP'
serial_port:str = '/dev/ttyS0'


def open_port(port: str) -> Union[None, serial.Serial]:
    s = None
    try:
        s = serial.Serial(port, 9600, timeout=0.05)
    except:
        msg = f"Could not open COM port {port}"
        print(msg)
    return s


def wait_for_start(s:serial.Serial, start_cmd:str, timeout_s: int=60) -> bool:
    cmd = []
    success = False
    bytes_to_read = len(start_cmd)
    sleep_interval_s = 0.1
    max_tries = timeout_s / sleep_interval_s
    
    while len(cmd) < 1 and max_tries > 0:
        cmd = s.read(bytes_to_read)
        # @todo: check if cmd matches start_cmd
        if cmd == start_cmd:
            success = True
            break
        time.sleep(0.1)
    
    if not success:
        print(f"wait_for_start timed out after {timeout_s} seconds")
    
    return success


if __name__ == "__main__":
    s = open_port(serial_port)
    print(s)
    if s is not None:
        wait_for_start(s, start_cmd=start_cmd)
