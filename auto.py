import torch
import torch.nn as nn
import os
import numpy as np
import subprocess
from collections import OrderedDict
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", default=579, type=int)
submitter = 'run.sh'

workdir = os.getcwd()+'/work/tool/OpenKE'
ck_dir = workdir+'/checkpoint_test'
step = int(80)

# extract the current max epoch on record
arr = os.listdir(workdir+'/checkpoint/')
num = []
for n in arr:
    if 'mid' in n:
        n = n.split('.')[0]
        num.append(n.split('id')[1]) # append number
maxi = sorted(num)[-1] # the max epoch

epoch = maxi+step

# for i in range(5):


# cannot use job array #SBATCH --array=x-y
# can exceed time limit of 2 hours

# load intermediate test ck and transform to transe model format
ck = torch.load('{}/ck300epoch79.ckpt'.format(ck_dir)) # always load the last ck in test folder
new_dict = OrderedDict()
new_dict['zero_const'] = ck['model.zero_const']
new_dict['pi_const'] = ck['model.pi_const']
new_dict['ent_embeddings.weight'] = ck['model.ent_embeddings.weight']
new_dict['rel_embeddings.weight'] = ck['model.rel_embeddings.weight']
torch.save(new_dict, '{}/checkpoint/mid{}.ckpt'.format(workdir, epoch)) # save formal ck every 80 epochs

# re-submit job 
# bash_command = 'sbatch {} {}'.format(submitter, epoch)
# process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
print('epoch {} checkpoint saved'.format(epoch))


# epoch += step




