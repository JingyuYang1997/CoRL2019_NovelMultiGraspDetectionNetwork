import serial
import pdb


dev1 = '/dev/ttyUSB0'
ser1 = serial.Serial(dev1, 115200)
print('open {}'.format(dev1))

def grasp2(angle):
    if angle>180:
        angle = 180
        print('over big, set to 180')
    if angle >= 100:
        angle_bytes = 'a0'+str(angle)
        angle_bytes = angle_bytes.encode()
    else:
        angle_bytes = 'a00'+str(angle)
        angle_bytes = angle_bytes.encode()
    ser1.write(angle_bytes)
    #print('grasp2 {} angle'.format(angle))

