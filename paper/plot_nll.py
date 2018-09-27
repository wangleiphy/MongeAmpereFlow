import numpy as np
import matplotlib.pyplot as plt 
from config import * 
import argparse 

parser = argparse.ArgumentParser(description='')

parser.add_argument("-filename", default='data/learn_mnist/Simple_MLP_hdim1024_Batchsize100_lr0.001_Nsteps100_epsilon0.1.log', help="filename")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")
args = parser.parse_args()

epoch, train_nll, test_nll = np.loadtxt(args.filename, unpack=True, usecols=(0,1,3))
#trace back only to the tail since the file is appended for many rounds
tail = int(epoch[-1]+1)
print (tail)
print (epoch[-tail:])
print (train_nll[-tail:])
#print (test_nll)

plt.figure(figsize=(4, 5))
plt.plot(epoch[-tail:], train_nll[-tail:], lw=2, zorder=99)
plt.plot(epoch[-tail:], test_nll[-tail:], lw=2, zorder=100)

print (np.mean(test_nll[-10:]), np.std(test_nll[-10:]))

plt.xlabel('epochs')
plt.ylabel('$\mathrm{NLL}$')
plt.xlim([1, 600])
plt.ylim([1250,1400])

plt.axhline(1380.8, color='r', lw=1, label='MADE')
plt.axhline(1323.2, color='g', lw=1, label='Real NVP')
plt.axhline(1300.5, color='b', lw=1, label='MAF')
plt.legend(loc='lower left')
plt.subplots_adjust(left=0.22, bottom=0.15)
plt.gca().set_xscale("log", nonposx='clip')

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)
