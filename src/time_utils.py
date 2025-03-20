"""
Provides:  
1. A contextmanager `time_to_exec` that can be used to measure the execution time of a task (must provide the name of the task).  
1. A dictionary `time_dict` that stores all the sum of the execution time for each task.  
1. A `print_time_dict` function to print all the execution times on te CLI.  
"""
import time
from contextlib import contextmanager

time_dict = {}

def print_time_dict():
    """
    Prints all the "{message}: {execution_time}" key/value pairs in time_dict.
    """
    global time_dict
    for msg, execution_time in time_dict.items():
        print(f"{msg}: {execution_time}s")

@contextmanager
def time_to_exec(message:str, print_msg=False):
    """
    ### Description:
    Measures the time to execute a code block.  
    The execution duration is stored in the global `time_dict` variable.  
    ### Args:
    - message: "Name" of the code block that will be used as a key in `time_dict`.
    - print_msg: If True, prints the execution time write away preceeded by the message.
    """
    global time_dict
    
    start_time = time.time()  # Record the start time
    yield  # Yield control to the block of code
    end_time = time.time()  # Record the end time after the block has executed
    execution_time = end_time - start_time  # Calculate the execution time
        
    if print_msg:
        print(f"{message}: {execution_time}s")  # Print the execution time with the message

    if not message in time_dict:
        time_dict[message] = execution_time
    else:
        time_dict[message] += execution_time

def reset_time_dict():
    global time_dict
    time_dict = {}