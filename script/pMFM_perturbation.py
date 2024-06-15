import os
import numpy as np
import time
import torch
import CBIG_pMFM_basic_functions_main as fc
import warnings

def CBIG_mfm_validation_desikan_main(gpu_index=0):
    '''
    This function is to validate the estimated parameters of mean field
    model.
    The objective function is the summation of FC correlation cost and
    FCD KS statistics cost.

    Args:
        gpu_index:      index of gpu used for optimization
    Returns:
        None
    '''

    # Setting GPU
    torch.cuda.set_device(gpu_index)

    # Create output folder
    input_path = 'F:/Thomas Yeo 代码/stable_projects/fMRI_dynamics/Kong2021_pMFM\output/step2_validation_results/'
    output_path = 'F:/fmridymamics_my/simulated_activity/'

    sc_mat_raw = fc.csv_matrix_read('../../input/Desikan_input/sc_train.csv')
    sc_mat = sc_mat_raw / sc_mat_raw.max() * 0.2
    sc_mat = torch.from_numpy(sc_mat).type(torch.FloatTensor).cuda()
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Load files
    n_node = 68
    vali_raw_all = np.zeros((4 * n_node + 1 + 8, 1))
    for i in range(6, 11):
        load_file = 'random_initialization_' + str(i) + '.csv'
        load_path = os.path.join(input_path, load_file)
        xmin = fc.csv_matrix_read(load_path)
        index_mat = np.zeros((2, xmin.shape[1]))
        index_mat[0, :] = i
        index_mat[1, :] = np.arange(xmin.shape[1])
        xmin = np.concatenate((index_mat, xmin), axis=0)
        vali_raw_all = np.concatenate((vali_raw_all, xmin), axis=1)
    vali_raw_all = vali_raw_all[:, 1:]
    index = np.argsort(vali_raw_all[7, :])
    sort_all = vali_raw_all[:, index]


    n_dup = 10
    parameter = np.zeros((4 * n_node + 4, 200))
    parameter1 = vali_raw_all[8:, 0:200]
    parameter[0:parameter1.shape[0], :] = parameter1.reshape(4 * n_node + 1,200)

    parameter = torch.from_numpy(parameter).type(torch.FloatTensor).cuda()
    n_num = parameter.shape[1]
    n_trial = 1
    result_save = np.zeros((parameter.shape[1]*3, n_trial))
    inter = np.zeros((6, 1))
    countloop = 0


    bold_d, S_E_d, S_I_d = fc.CBIG_mfm_multi_simulation_output(parameter, sc_mat, 14.4, n_dup,0.1, 0.05)
    FC_sim = fc.CBIG_FCcorrelation_multi_simulation_output(bold_d, n_dup)
    #FC_sim = FC_sim.cpu().numpy()

    results = np.zeros((n_node, 3600+n_node))
    for i in range(n_num):
        savefile = ['simulation_parameter_AI_L_0.1_I_R_0.05_E_', str(i), '.csv']
        S_I_sim = torch.zeros(n_node, 1200)
        S_E_sim = torch.zeros(n_node, 1200)
        bold_sim = torch.zeros(n_node, 1200)
        for j in range(n_dup):
            S_I_sim = S_I_sim + torch.squeeze(S_I_d[:,(j-1)*n_num+i, :])
            S_E_sim = S_E_sim + torch.squeeze(S_E_d[:, (j - 1) * n_num + i, :])
            bold_sim = bold_sim + torch.squeeze(bold_d[:, (j - 1) * n_num + i, :])
        S_I_sim = S_I_sim / n_dup
        S_E_sim = S_E_sim / n_dup
        bold_sim = bold_sim / n_dup
        bold_sim = bold_sim.cpu().numpy()
        S_E_sim = S_E_sim.cpu().numpy()
        S_I_sim = S_I_sim.cpu().numpy()
        results[:,0:1200] = bold_sim
        results[:,1200:2400] = S_E_sim
        results[:, 2400:3600] = S_I_sim
        results[:, 3600:3700] = torch.squeeze(FC_sim[:,:,i]).cpu().numpy()
        save_path = [output_path] + savefile
        np.savetxt(''.join(save_path), results, delimiter=',')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    CBIG_mfm_validation_desikan_main()