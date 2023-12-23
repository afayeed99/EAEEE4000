# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:40:05 2023

@author: Chris
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob

import h5py
import scipy.io

#from utils import * 


print('done')
#%%


ori_path = os.getcwd()

fileout = ['Batch1_', 'Batch2_', 'Batch3_']
filein = ['2017-05-12_batchdata_updated_struct_errorcorrect', '2017-06-30_batchdata_updated_struct_errorcorrect', '2018-04-12_batchdata_updated_struct_errorcorrect']



for file_idx in [0, 1 ,2]:

    filename = ori_path + '\\Paper 1\\' + filein[file_idx]+ '.mat'
    file = h5py.File(filename)

    batch_summary_file = file['batch']['summary'] # batch_summary_file[i,j], where j = 0, and i = 0 to 45
    num_cells = batch_summary_file.shape[0]
    
    cycle_life_file = file['batch']['cycle_life'] # summary of cycle life
    
    cycles_file = file['batch']['cycles'] # data per cycles
    
    summ_temp = [] #temporary list for summary arrays
    cell_id = []
    summ_dat = []
    cell_CC = []
    ind = 0
    for i in range(num_cells):
        if (file_idx != 1 or i not in [1, 7, 8, 9, 15, 16]) and (file_idx != 2 or i not in [37]):  #ignore bad cells in Batches 2 and 3
            cycle_life = file[cycle_life_file[i,0]][0]
        
            internalR_data = file[batch_summary_file[i,0]]['IR'][0,:].tolist()
            charge_data = file[batch_summary_file[i,0]]['QCharge'][0,:].tolist()
            discharge_data = file[batch_summary_file[i,0]]['QDischarge'][0,:].tolist()
            tavg_data = file[batch_summary_file[i,0]]['Tavg'][0,:].tolist()
            tmin_data = file[batch_summary_file[i,0]]['Tmin'][0,:].tolist()
            tmax_data = file[batch_summary_file[i,0]]['Tmax'][0,:].tolist()
            chargetime_data = file[batch_summary_file[i,0]]['chargetime'][0,:].tolist()
            cycleNum_data = file[batch_summary_file[i,0]]['cycle'][0,:].tolist()
                    
        
            ################ SUMMARY DATA FOR EACH CELL INTO A FORMATTED FILE FOR ML #############    
            summary_IR = np.hstack(internalR_data)
            shape = [len(summary_IR),1]
            summary_IR.reshape(shape)
            summary_charge = np.hstack(charge_data)
            summary_discharge = np.hstack(discharge_data)
            summary_tavg = np.hstack(tavg_data)
            summary_tmin = np.hstack(tmin_data)
            summary_tmax = np.hstack(tmax_data)
            summary_chargetime = np.hstack(chargetime_data)
            summary_cycleNum = np.hstack(cycleNum_data)
            
            cell_num = np.ones(shape=len(summary_IR))*i
             
            print('Done with cell ' + str(i))
        
            ################ DATA PER CYCLE FOR EACH CELL INTO A FORMATTED FILE FOR ML #############
            cycles = file[cycles_file[i,0]]
            
            cycle_temp = [] #temporary list for cycle arrays
            CC_dat = [] #list for storing charge currents for each cycle
            
            current = np.array((file[cycles['I'][5,0]])).squeeze()
            vals, counts = np.unique(np.around(current[current>1],decimals=1), return_counts=True)
            order = np.argsort(counts)[::-1]
            ordered = vals[order]
            CC = ordered[0:3]
            CC = np.sort(CC)[::-1]
            #get max of IR, Tavg, Tmax for first 100 cycles
            IrMax = summary_IR[0:99].max()
            TavMax = summary_tavg[0:99].max()
            TmMax = summary_tmax[0:99].max()
            summ_dat.append(np.hstack([IrMax,TavMax,TmMax]))
            
            ds = xr.Dataset({'IR':   IrMax,
                             'Tavg': TavMax,
                             'Tmax': TmMax,
                             'I1': CC[0],
                             'I2': CC[1],
                             'I3': CC[2],}                   
                            )
            ds.expand_dims(dim={'cell_id': i})
            summ_temp.append(ds)
            
            
    combined_summ = xr.concat(summ_temp, dim='cell_id')

    combined_summ.to_netcdf(path=fileout[file_idx]+'_TempsAndCurrents.nc', mode='w')

#%%

