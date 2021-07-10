
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
        aux.append(analogInput_CE1(2,spi2))
        #print('from the CE0_C1: ',output)
        aux.append(analogInput_CE0(3,spi))
        #prin(('from the CE0_C2: ',output)
        aux.append(analogInput_CE0(7,spi))
        #print('from the CE0_C3: ',output)
        aux.append(analogInput_CE1(3,spi2))
        #print('from the CE0_C4: ',output)
        aux.append(analogInput_CE1(1,spi2))
        #print('from the CE1_C1: ',output)
        aux.append(analogInput_CE0(5,spi))
        #print('from the CE1_C2: ',output)
        aux.append(analogInput_CE0(1,spi))
        #print('from the CE1_C3: ',output)
        aux.append(analogInput_CE0(2,spi))
        #print('from the CE1_C4: ',output)
        aux.append(analogInput_CE0(6,spi))
        if len(output) % 500 == 0:
		print('Data: ',aux)
	output.append(aux)
	#termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
	#fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        #threading.Timer(0.1,read_data).start()

def write_into_file():
        try:
                inkey = sys.stdin.read(1)
                if inkey == 'n':
                        print('start to store data')
                        with open('Training_data2/9/data_9_221_s4_t1_3_v1.csv', 'w') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(['time', '6','1','2','3','9','8','7','4','5'])
                                csvwriter.writerows(output)
			print('End store data')
        except IOError: pass

def isCovered(data, digit):
    if data[data.digit < 150].count() > 170:
        return True
    else: 
        return False

def calibration_mode():
    print("Calibration Mode: Please put your finger tip on the center photodiode  ")
    data_cal = pd.DataFrame(output[300:500], columns = ['6', '1', '2', '3', '9', '8', '7', '4', '5'])
    if isCovered(data_cal, 2):
        print ("sensing area is : 3; data_reading area: 1" )
    elif isCovered(data_cal, 1):
        print ("sensing area is : 4; data_reading area: 1" )     
    elif isCovered(data_cal, 3):
        print ("sensing area is : 3; data_reading area: 2" )
    elif isCovered(data_cal, 4) and (not isCovered(data_cal, 5) ):
        print ("sensing area is : 2; data_reading area: 1" )   
    elif isCovered(data_cal, 6) and (not isCovered(data_cal, 5) ):
        print ("sensing area is : 1; data_reading area: 2" )
    elif isCovered(data_cal, 8) and (not isCovered(data_cal, 5) ):
        print ("sensing area is : 1; data_reading area: 3" )
    elif isCovered(data_cal, 7) and (not isCovered(data_cal, 5) ):
        print ("sensing area is : 2; data_reading area: 3" )
    elif isCovered(data_cal, 9) and (not isCovered(data_cal, 5) ):
        print ("sensing area is : 1; data_reading area: 4" )
    else:
        print ("sensing area is : 1; data_reading area: 1" )   
    return True		

		
		
if __name__=='__main__':
	spi_init()
	termios_init()
	RepeatedTimer(0.01,read_data).start()
	calibration_mode()
	
	try:
		while 1:
			write_into_file()
		#read_data()
		#RepeatedTimer(0.1,read_data).start()
        finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

	#threading.Timer(0.1,read_data).start()