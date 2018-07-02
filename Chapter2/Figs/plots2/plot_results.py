import matplotlib.pyplot as plt
import csv

t = []
seg_loss = []
seg_var = []
inst_loss = []
inst_var = []
depth_loss = []
depth_var = []

with open('fullsize_experiment_results.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    header = True
    for row in plots:
        if header:
            header = False
            continue
        t.append(float(row[0]))
        seg_loss.append(float(row[1]))
        seg_var.append(float(row[3]))
        depth_loss.append(float(row[4]))
        depth_var.append(float(row[6]))
        inst_loss.append(float(row[7]))
        inst_var.append(float(row[9]))

plt.plot(t,seg_loss, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
#plt.legend()
#plt.show()
plt.savefig('segmentation_loss.eps', format='eps')
plt.close() 

plt.plot(t,inst_loss, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.savefig('instance_loss.eps', format='eps')
plt.close() 

plt.plot(t,depth_loss, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.savefig('depth_loss.eps', format='eps')
plt.close() 

# variances
plt.plot(t,seg_var, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel(r'Task Uncertainty ($\sigma^2$)')
#plt.legend()
#plt.show()
plt.savefig('segmentation_var.eps', format='eps')
plt.close() 

plt.plot(t,inst_var, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel(r'Task Uncertainty ($\sigma^2$)')
plt.savefig('instance_var.eps', format='eps')
plt.close() 

plt.plot(t,depth_var, label='Loss')
plt.xlabel('Training Iterations')
plt.ylabel(r'Task Uncertainty ($\sigma^2$)')
plt.savefig('depth_var.eps', format='eps')
plt.close() 


import numpy as np
seg_var = np.array(seg_var)*10
sum = 1 / np.array(seg_var) + 1 / np.array(depth_var) + 1 / np.array(inst_var)
sum = 1/sum
seg_var_n = sum / np.array(seg_var) 
inst_var_n = sum / np.array(inst_var) 
depth_var_n = sum / np.array(depth_var) 

# Make the plot
plt.stackplot(t,  seg_var_n,  inst_var_n,  depth_var_n, labels=['segmentation','instance','depth'])
plt.legend(loc='upper left')
plt.show()
