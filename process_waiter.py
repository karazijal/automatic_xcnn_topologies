import os, errno, sys, time

def is_running(pid):
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
    return True

def main(): # a litte util to queue for GPU on the server
    if len(sys.argv) <=1:
        exit(-1)
    pid = int(sys.argv[1])
    # pid = 37142
    while is_running(pid):
        time.sleep(2)

if __name__=="__main__":
    main()

