
import spidev
from numpy import interp
from time import sleep
import RPi.GPIO as GPIO
import signal
import datetime, time
from threading import _Timer
import csv
import termios,sys,tty, fcntl, os,select
import threading

global spi, spi2, output,fd, oldterm, newattr, oldflags
spi = spidev.SpiDev()
spi2 = spidev.SpiDev()
output = []
fd = sys.stdin.fileno()
oldterm = termios.tcgetattr(fd)
oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
newattr = termios.tcgetattr(fd)

'''
link: https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds/25251804#25251804
'''
class RepeatedTimer(object):
  def __init__(self, interval, function, *args, **kwargs):
    self._timer = None
    self.interval = interval
    self.function = function
    self.args = args
    self.kwargs = kwargs
    self.is_running = False
    self.next_call = time.time()
    self.start()

  def _run(self):
    self.is_running = False
    self.start()
    self.function(*self.args, **self.kwargs)

  def start(self):
    if not self.is_running:
      self.next_call += self.interval
      self._timer = threading.Timer(self.next_call - time.time(), self._run)
      self._timer.start()
      self.is_running = True

  def stop(self):
    self._timer.cancel()
    self.is_running = False

'''
System RepeatingTimer
link: https://juejin.im/post/6844903796254965768 
'''
class RepeatTimer(_Timer):
	def run(self):
		while not self.finished.is_set():
			self.function(*self.args, **self.kwargs)
			self.finished.wait(self.interval)


'''
The termios with nonBlock 
link: https://docs.python.org/2/faq/library.html#how-do-i-get-a-single-keypress-at-a-time
link: https://shallowsky.com/blog/programming/python-read-characters.html
'''
def termios_init():
	#newattr = termios.tcgetattr(fd)
	newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
	termios.tcsetattr(fd, termios.TCSANOW, newattr)

	#oldterm = termios.tcgetattr(fd)
	#oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
	fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def analogInput_CE0(channel,spi):
    adc = spi.xfer2([1, (8+channel)<<4, 0])
    data = ((adc[1]&3) <<8) + adc[2]
    return data

def analogInput_CE1(channel,spi2):
    adc = spi2.xfer2([1, (8+channel)<<4, 0])
    data = ((adc[1]&3) <<8) + adc[2]
    return data

def spi_init():
        spi.open(0,0)
        spi.max_speed_hz =1350000
        spi2.open(0,1)
        spi2.max_speed_hz =1350000

def read_data():
	aux = []
	#inp, outp, err = select.select([sys.stdin],[],[])
	'''try:
		inkey = sys.stdin.read(1)
		if inkey == 'n':
			with open('data_1', 'w') as csvfile:
				csvwriter = csv.writer(csvfile)
    				csvwriter.writerow(['time', 'CE0_C1','CE0_C2','CE0_C3','CE0_C4','CE1_C1','CE1_C2','CE1_C3','CE1_C4','CE1_C5'])
    				csvwriter.writerows(output)
	except IOError: pass
	finally:
		termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    		fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)'''
	aux.append(time.time())
	#print("start time is: ",time.time() )
        aux.append(analogInput_CE0(0,spi))
        #print('from the CE0_C1: ',output)
        aux.append(analogInput_CE0(1,spi))
        #prin(('from the CE0_C2: ',output)
        aux.append(analogInput_CE0(2,spi))
        #print('from the CE0_C3: ',output)
        aux.append(analogInput_CE0(3,spi))
        #print('from the CE0_C4: ',output)
        aux.append(analogInput_CE1(0,spi2))
        #print('from the CE1_C1: ',output)
        aux.append(analogInput_CE1(1,spi2))
        #print('from the CE1_C2: ',output)
        aux.append(analogInput_CE1(2,spi2))
        #print('from the CE1_C3: ',output)
        aux.append(analogInput_CE1(3,spi2))
        #print('from the CE1_C4: ',output)
        aux.append(analogInput_CE1(4,spi2))
        print('data : ',aux)
	output.append(aux)
	#termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
	#fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        #threading.Timer(0.1,read_data).start()

def write_into_file():
        try:
                inkey = sys.stdin.read(1)
                if inkey == 'n':
                        with open('data_1_v6.csv', 'w') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(['time', 'CE0_C1','CE0_C2','CE0_C3','CE0_C4','CE1_C1','CE1_C2','CE1_C3','CE1_C4','CE1_C5'])
                                csvwriter.writerows(output)
        except IOError: pass


if __name__=='__main__':
	spi_init()
	termios_init()
	RepeatedTimer(0.05,read_data).start()

	try:
		while 1:
			write_into_file()
		#read_data()
		#RepeatedTimer(0.1,read_data).start()
        finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

	#threading.Timer(0.1,read_data).start()
