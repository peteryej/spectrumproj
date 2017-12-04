import channel_generator as cg
import time
import os

START = 0
END = 1
WINDOW = 10
DATA_FOLDER = 'multi_channel_data'
DATA_FILE = 'data'
INPUTFILE = 'inputmultitraining.txt'

def parse_a_line(line):
    info_list = []
    items = line.split(';')
    for i in range(4):
        if items[i].isspace() or (not items[i]):
            info_list.append(None)
        else:
            its = items[i].split(',')
            d = {}
            d['start'] = float(its[0].strip())
            d['cf'] = int(its[1].strip())
            d['bw'] = int(its[2].strip())
            d['mod'] = its[3].strip()[:3]
            d['mod_num'] = int(its[3].strip()[3:])
            d['duration'] = float(its[4].strip())
            info_list.append(d)
    return info_list

def create_time_line(info_list):
    l = []
    for i, info in enumerate(info_list):
        if info is None:
            continue
        l.append((info['start'], i+1, START))
        l.append((info['start']+info['duration'], i+1, END))

    return sorted(l, key = lambda item: item[0])



def read_one_transmitter(para, i, sig):
    '''
    sig: (cf, bw, mod_num)
    '''
    si = str(i)
    #para['on_off'+si] = 0
    para['cf'+si] = sig[0]
    para['bw'+si] = sig[1]
    if i <= 2:
        para['PSKMod'+si] = sig[2]
    else:
        para['QAMMod'+si] = sig[2]


fake = [(910,5,2),(915,2,8),(920,4,16),(925,7,64)]    

def create_parameters(info_list):
    para = {}
    for i, info in enumerate(info_list):
        if info is None:
            continue
        read_one_transmitter(para, i+1, (info['cf'], info['bw'], info['mod_num']))
    return para

def start_simulate(para, timeline):
    tb = cg.channel_generator(**para)
    tb.start()
    time.sleep(timeline[0][0])
    for i, item in enumerate(timeline):
        id = item[1]
        if id == 1:
            f = tb.set_on_off1
        elif id == 2:
            f = tb.set_on_off2
        elif id == 3:
            f = tb.set_on_off3
        else:
            f = tb.set_on_off4

        if item[2] == START:
            f(1)
        else:
            f(0)

        if i < len(timeline) - 1:
            time.sleep(timeline[i+1][0] - timeline[i][0])
        else:
            time.sleep(WINDOW - timeline[i][0])

    tb.stop()

def main_routine(line, num):
    info_list = parse_a_line(line)
    if not any(info_list):
        return
    para = create_parameters(info_list) 
    para['file_name'] = os.path.join(DATA_FOLDER, DATA_FILE+str(num)+'.dat')
    timeline = create_time_line(info_list)
    print 'outputing file: ' + para['file_name']
    start_simulate(para, timeline)




        

if __name__ == '__main__':
    with open(INPUTFILE) as input:
        for i, line in enumerate(input):
            main_routine(line, i)
