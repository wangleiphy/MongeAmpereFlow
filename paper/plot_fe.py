import numpy as np
import matplotlib.pyplot as plt 
from config import * 
import argparse 

parser = argparse.ArgumentParser(description='')

parser.add_argument("-filename", default='data/learn_ot/ising_L16_d2_T2.269185314213022_symmetrize_Simple_MLP_hdim512_Batchsize64_lr0.001_delta0.0_Nsteps50_epsilon0.1.log', help="filename")

parser.add_argument("-fe_exact",type=float,default=-2.3159198563359373, help="fe_exact")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")
args = parser.parse_args()


epoch, fe = np.loadtxt(args.filename, unpack=True, usecols=(0,1))
tail = int(epoch[-1]+1)

plt.figure(figsize=(8, 5)) 
plt.plot(epoch[-tail:], fe[-tail:])
plt.axhline(args.fe_exact, color='r', lw=2)
#plt.plot(epoch, (fe-args.fe_exact)/abs(args.fe_exact))
plt.xlabel('epochs')
plt.xlim([0, 2000])

#plt.legend()
plt.subplots_adjust(bottom=0.15)
#plt.gca().set_yscale("log")    
plt.ylabel('$\mathcal{L}/N$' )

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)
