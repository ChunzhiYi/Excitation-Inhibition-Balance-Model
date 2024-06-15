
import os
import numpy as np
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
    input_path = 'F:/Thomas Yeo 代码/stable_projects/fMRI_dynamics/Kong2021_pMFM/output/step3_test_results/'
    output_path = 'F:/fmridymamics_my/WM_atlas_DA/WM_results/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Load files
    load_file = 'test_all.csv'
    load_path = os.path.join(input_path, load_file)
    xmin = fc.csv_matrix_read(load_path)
    n_node = 68
    n_dup = 15
    parameter = np.zeros((4 * n_node + 4, 10))
    parameter1 = xmin[11:, :]
    parameter[0:parameter1.shape[0], :] = parameter1.reshape(4 * n_node + 1, 10)

    n_trial = 10
    result_save = np.zeros((parameter.shape[1]*3, n_trial))
    inter = np.zeros((6, 1))
    countloop = 0

    for j in range(n_trial):
        vali_total, vali_corr, vali_ks = \
            fc.CBIG_combined_cost_validation_WM(
                parameter, n_dup)
        result_save[0:10, j] = vali_total
        result_save[10:20, j] = vali_corr
        result_save[20:30, j] = vali_ks
    # result_save = result_save[:,1:]
    test_file_all = os.path.join(output_path, 'WM_vali.csv')
    np.savetxt(test_file_all, result_save, delimiter=',')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    CBIG_mfm_validation_desikan_main()
