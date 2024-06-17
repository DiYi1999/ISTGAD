import pandas as pd
import pickle
import numpy as np
import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager
from matplotlib.backends.backend_pdf import PdfPages
import scienceplots
import latex


"""
Some Details:
1.  Thanks for: https://www.mdpi.com/2076-3417/12/17/8634.
2.  We do not possess the rights to redistribute others' data; 
    hence, you could download the original data from https://github.com/BIRDSOpenSource/EPS_dataset. 
    We provide only our data preprocessing code here.
3.  The original data we utilized is sourced from the BIRDS satellite constellation, 
    which comprises four satellites: TSURU, UGUISU, NEPALISAT, and RAAVANA.
4.  The TSURU and UGUISU satellites exhibit no anomalies. 
    whereas NEPALISAT and RAAVANA exhibit significant anomalies that occupy a large proportion.
5.  The original sampling frequencies of the four datasets differ. 
    If necessary, readers can implement downsampling steps for RAAVANA and UGUISU. 
    However, due to the limited data volume, downsampling may lead to information loss. 
    In fact, it is a common phenomenon in the spacecraft field that sampling frequencies are inconsistent, 
    a challenge we must overcome. 
    It is recommended that the developed models should be robust against such inconsistencies, 
    and this dataset can effectively verify this robustness.
6.  About the feature selection, we offer two versions: BIRDS_6535part_10sensor and BIRDS_6535part, 
    readers can choose according to their needs.
7.  We hope that our data analysis code can help you better understand this dataset. 
    Good Luck!
"""


# In[00]:

# plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.style.use('science')

def anomaly_and_observation_error(data_Feb):

    data_Feb_all_label = data_Feb.copy()
    data_Feb_all_label.loc[:, :] = 0

    data_Feb_all_label1 = data_Feb_all_label.copy()

    data_Feb_all_label2 = data_Feb_all_label.copy()



    data_Feb_all_label.loc[data_Feb['Vpy (mV)'] < 1200, ['Vpy (mV)', 'Ipy (mA)']] = 1
    data_Feb_all_label.loc[data_Feb['Vpx (mV)'] < 1200, ['Vpx (mV)', 'Ipx (mA)']] = 1
    data_Feb_all_label.loc[data_Feb['Vmz (mV)'] < 1200, ['Vmz (mV)', 'Imz (mA)']] = 1
    data_Feb_all_label.loc[data_Feb['Vmx (mV)'] < 1200, ['Vmx (mV)', 'Imx (mA)']] = 1
    data_Feb_all_label.loc[data_Feb['Vpz (mV)'] < 1200, ['Vpz (mV)', 'Ipz (mA)']] = 1

    data_Feb_all_label1.loc[data_Feb['Vpy (mV)'] < 1200, ['Vpy (mV)', 'Ipy (mA)']] = 1
    data_Feb_all_label1.loc[data_Feb['Vpx (mV)'] < 1200, ['Vpx (mV)', 'Ipx (mA)']] = 1
    data_Feb_all_label1.loc[data_Feb['Vmz (mV)'] < 1200, ['Vmz (mV)', 'Imz (mA)']] = 1
    data_Feb_all_label1.loc[data_Feb['Vmx (mV)'] < 1200, ['Vmx (mV)', 'Imx (mA)']] = 1
    data_Feb_all_label1.loc[data_Feb['Vpz (mV)'] < 1200, ['Vpz (mV)', 'Ipz (mA)']] = 1


    data_Feb_all_label.loc[((data_Feb['Vpy (mV)'] >= 2000) &
                            (data_Feb['Vpy (mV)'] <= 3000) &
                            (data_Feb['Ipy (mA)'] < 50)),
                           ['Vpy (mV)', 'Ipy (mA)']] = 1
    data_Feb_all_label.loc[((data_Feb['Vpx (mV)'] >= 2000) &
                            (data_Feb['Vpx (mV)'] <= 3000) &
                            (data_Feb['Ipx (mA)'] < 50)),
                           ['Vpx (mV)', 'Ipx (mA)']] = 1
    data_Feb_all_label.loc[((data_Feb['Vmz (mV)'] >= 2000) &
                            (data_Feb['Vmz (mV)'] <= 3000) &
                            (data_Feb['Imz (mA)'] < 50)),
                           ['Vmz (mV)', 'Imz (mA)']] = 1
    data_Feb_all_label.loc[((data_Feb['Vmx (mV)'] >= 2000) &
                            (data_Feb['Vmx (mV)'] <= 3000) &
                            (data_Feb['Imx (mA)'] < 50)),
                           ['Vmx (mV)', 'Imx (mA)']] = 1
    data_Feb_all_label.loc[((data_Feb['Vpz (mV)'] >= 2000) &
                            (data_Feb['Vpz (mV)'] <= 3000) &
                            (data_Feb['Ipz (mA)'] < 50)),
                           ['Vpz (mV)', 'Ipz (mA)']] = 1

    data_Feb_all_label2.loc[((data_Feb['Vpy (mV)'] >= 2000) &
                             (data_Feb['Vpy (mV)'] <= 3000) &
                             (data_Feb['Ipy (mA)'] < 50)),
                            ['Vpy (mV)', 'Ipy (mA)']] = 1
    data_Feb_all_label2.loc[((data_Feb['Vpx (mV)'] >= 2000) &
                             (data_Feb['Vpx (mV)'] <= 3000) &
                             (data_Feb['Ipx (mA)'] < 50)),
                            ['Vpx (mV)', 'Ipx (mA)']] = 1
    data_Feb_all_label2.loc[((data_Feb['Vmz (mV)'] >= 2000) &
                             (data_Feb['Vmz (mV)'] <= 3000) &
                             (data_Feb['Imz (mA)'] < 50)),
                            ['Vmz (mV)', 'Imz (mA)']] = 1
    data_Feb_all_label2.loc[((data_Feb['Vmx (mV)'] >= 2000) &
                             (data_Feb['Vmx (mV)'] <= 3000) &
                             (data_Feb['Imx (mA)'] < 50)),
                            ['Vmx (mV)', 'Imx (mA)']] = 1
    data_Feb_all_label2.loc[((data_Feb['Vpz (mV)'] >= 2000) &
                             (data_Feb['Vpz (mV)'] <= 3000) &
                             (data_Feb['Ipz (mA)'] < 50)),
                            ['Vpz (mV)', 'Ipz (mA)']] = 1

    return data_Feb_all_label, data_Feb_all_label1, data_Feb_all_label2

def channel_plot(plot_dirname_path, data_Feb, data_Feb_all_label, data_Feb_all_label1, data_Feb_all_label2):
    pdf = PdfPages(plot_dirname_path)
    plt.rcParams['figure.figsize'] = 6, 1.5
    for dim in range(data_Feb.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(data_Feb.values[:, dim], linewidth=0.2, color='k')
        ax_1 = ax.twinx()
        ax_1.fill_between(np.arange(data_Feb.shape[0]), data_Feb_all_label1.values[:, dim], color='blue', alpha=0.3)
        ax_2 = ax.twinx()
        ax_2.fill_between(np.arange(data_Feb.shape[0]), data_Feb_all_label2.values[:, dim], color='red', alpha=0.3)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.set_ylabel(str(channel_list[dim]) + ' channel')
        # fig.tight_layout()
        # fig.tight_layout()是调整整体空白，使得各子图之间的空白更合理
        pdf.savefig(fig)
        plt.close()
    pdf.close()



channel_list = ['Tpy (°C)', 'Tpx (°C)', 'Tmz (°C)', 'Tmx (°C)', 'Tpz (°C)',
                'Vpy (mV)', 'Vpx (mV)', 'Vmz (mV)', 'Vmx (mV)', 'Vpz (mV)',
                'Ipy (mA)', 'Ipx (mA)', 'Imz (mA)', 'Imx (mA)', 'Ipz (mA)',
                'Vbat (V)', 'Ibatt(mA)', 'Tbatt (℃)']



data_RAAVANA_Feb = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/RAAVANA.xlsx',
                         sheet_name='13 Feb 2021')
data_RAAVANA_Feb = data_RAAVANA_Feb[channel_list]

data_RAAVANA_Feb_all_label, data_RAAVANA_Feb_all_label1, data_RAAVANA_Feb_all_label2 = anomaly_and_observation_error(data_RAAVANA_Feb)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/RAVANA_pro/RAAVANA_2021Feb13.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_RAAVANA_Feb, data_RAAVANA_Feb_all_label, data_RAAVANA_Feb_all_label1, data_RAAVANA_Feb_all_label2)
print("finish")



data_RAAVANA_Mar = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/RAAVANA.xlsx',
                         sheet_name='11 March 2021')
data_RAAVANA_Mar = data_RAAVANA_Mar[channel_list]

data_RAAVANA_Mar_all_label, data_RAAVANA_Mar_all_label1, data_RAAVANA_Mar_all_label2 = anomaly_and_observation_error(data_RAAVANA_Mar)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/RAVANA_pro/RAAVANA_2021March11.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_RAAVANA_Mar, data_RAAVANA_Mar_all_label, data_RAAVANA_Mar_all_label1, data_RAAVANA_Mar_all_label2)
print("finish")



data_UGUISU_Oct = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/UGUISU.xlsx',
                         sheet_name='23 October 2020')
data_UGUISU_Oct = data_UGUISU_Oct[channel_list]

data_UGUISU_Oct_all_label, data_UGUISU_Oct_all_label1, data_UGUISU_Oct_all_label2 = anomaly_and_observation_error(data_UGUISU_Oct)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/UGUISU_pro/UGUISU_2020Oct23.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_UGUISU_Oct, data_UGUISU_Oct_all_label, data_UGUISU_Oct_all_label1, data_UGUISU_Oct_all_label2)
print("finish")



data_UGUISU_Feb = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/UGUISU.xlsx',
                         sheet_name='13 Feb 2021')
data_UGUISU_Feb = data_UGUISU_Feb[channel_list]

data_UGUISU_Feb_all_label, data_UGUISU_Feb_all_label1, data_UGUISU_Feb_all_label2 = anomaly_and_observation_error(data_UGUISU_Feb)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/UGUISU_pro/UGUISU_2021Feb13.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_UGUISU_Feb, data_UGUISU_Feb_all_label, data_UGUISU_Feb_all_label1, data_UGUISU_Feb_all_label2)
print("finish")



data_UGUISU_Apr = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/UGUISU.xlsx',
                                sheet_name='10 April 2021')
data_UGUISU_Apr = data_UGUISU_Apr[channel_list]

data_UGUISU_Apr_all_label, data_UGUISU_Apr_all_label1, data_UGUISU_Apr_all_label2 = anomaly_and_observation_error(data_UGUISU_Apr)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/UGUISU_pro/UGUISU_2021Apr10.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_UGUISU_Apr, data_UGUISU_Apr_all_label, data_UGUISU_Apr_all_label1, data_UGUISU_Apr_all_label2)
print("finish")


data_TSURU = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/TSURU.xlsx',
                           sheet_name=None)

data_TSURU = pd.concat(data_TSURU.values(), ignore_index=True)
data_TSURU = data_TSURU[channel_list]
"""# 25653*18"""

data_TSURU_all_label, data_TSURU_all_label1, data_TSURU_all_label2 = anomaly_and_observation_error(data_TSURU)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/TSURU_pro/TSURU2.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_TSURU, data_TSURU_all_label, data_TSURU_all_label1, data_TSURU_all_label2)
print("finish")


data_NEPALISAT = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/NEPALISAT.xlsx',
                               sheet_name=None)

data_NEPALISAT = pd.concat(data_NEPALISAT.values(), ignore_index=True)
data_NEPALISAT = data_NEPALISAT[channel_list]
"""# 3040*18"""

data_NEPALISAT_all_label, data_NEPALISAT_all_label1, data_NEPALISAT_all_label2 = anomaly_and_observation_error(data_NEPALISAT)

plot_dirname_path = '/home/data/DATA/BIRDS_1U_CubeSat/NEPALISAT_pro/NEPALISAT.pdf'
dirname = os.path.dirname(plot_dirname_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
channel_plot(plot_dirname_path, data_NEPALISAT, data_NEPALISAT_all_label, data_NEPALISAT_all_label1, data_NEPALISAT_all_label2)
print("finish")



# In[01]:


# plt.style.reload_library()
plt.style.use('science')
# plt.style.use(['science', 'ieee'])
# plt.rcParams["text.usetex"] = False
# plt.rcParams["text.usetex"] = True
# plt.rcParams['font.family'] = ['serif']
# # plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']


def channel_plot(plot_dirname_path,
                 data,
                 channel_list=None,
                 data_all_label=None):
    pdf = PdfPages(plot_dirname_path)
    plt.rcParams['figure.figsize'] = data.shape[0] * 6/15000, 1
    for dim in range(data.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(data.values[:, dim], linewidth=0.2, color='k')
        if data_all_label is not None:
            if np.any(data_all_label.values[:, dim]):
                ax_2 = ax.twinx()
                ax_2.fill_between(np.arange(data.shape[0]), data_all_label.values[:, dim], color='red', alpha=0.3)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.set_ylabel(str(channel_list[dim]))
        # fig.tight_layout()
        pdf.savefig(fig)
        plt.close()
    pdf.close()

# 通道名
channel_list = ['Tpy (°C)', 'Tpx (°C)', 'Tmz (°C)', 'Tmx (°C)', 'Tpz (°C)',
                'Vpy (mV)', 'Vpx (mV)', 'Vmz (mV)', 'Vmx (mV)', 'Vpz (mV)',
                'Ipy (mA)', 'Ipx (mA)', 'Imz (mA)', 'Imx (mA)', 'Ipz (mA)',
                'Vbat (V)', 'Ibatt(mA)', 'Tbatt (℃)']




data_TSURU = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/TSURU.xlsx',
                            sheet_name=None)
data_TSURU.pop('Test1 w batt',None)
data_TSURU.pop('Test2 wo batt',None)
data_TSURU = pd.concat(data_TSURU.values(), ignore_index=True)
data_TSURU = data_TSURU[channel_list]
"""# 24541*18"""

data_NEPALISAT = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/NEPALISAT.xlsx',
                               sheet_name=None)
data_NEPALISAT = pd.concat(data_NEPALISAT.values(), ignore_index=True)
data_NEPALISAT = data_NEPALISAT[channel_list]
"""# 3040*18"""

data_RAAVANA = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/RAAVANA.xlsx',
                             sheet_name=None)
data_RAAVANA = pd.concat(data_RAAVANA.values(), ignore_index=True)
data_RAAVANA = data_RAAVANA[channel_list]
"""# 2159*18"""

data_UGUISU = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/UGUISU.xlsx',
                            sheet_name=None)
data_UGUISU = pd.concat(data_UGUISU.values(), ignore_index=True)
data_UGUISU = data_UGUISU[channel_list]
"""# 3270*18"""

# label:
# The first type of anomaly comes from RAAVANA
# detail: https://www.mdpi.com/2076-3417/12/17/8634
data_RAAVANA_all_label = data_RAAVANA.copy()
data_RAAVANA_all_label.loc[:, :] = 0
data_RAAVANA_all_label.loc[data_RAAVANA['Vpy (mV)'] < 1200, ['Vpy (mV)', 'Ipy (mA)']] = 1

# The second type of anomaly comes UGUISU
# detail: https://www.mdpi.com/2076-3417/12/17/8634
data_UGUISU_all_label = data_UGUISU.copy()
data_UGUISU_all_label.loc[:, :] = 0
data_UGUISU_all_label.loc[((data_UGUISU['Vpy (mV)'] >= 2000) &
                           (data_UGUISU['Vpy (mV)'] <= 3000) &
                           (data_UGUISU['Ipy (mA)'] < 50)),
                          ['Vpy (mV)', 'Ipy (mA)']] = 1

# TSURU
data_TSURU_all_label = data_TSURU.copy()
data_TSURU_all_label.loc[:, :] = 0

# NEPALISAT
data_NEPALISAT_all_label = data_NEPALISAT.copy()
data_NEPALISAT_all_label.loc[:, :] = 0

# spilt
data_all = pd.concat([data_TSURU, data_NEPALISAT, data_RAAVANA, data_UGUISU], axis=0, ignore_index=True)
data_all_label = pd.concat([data_TSURU_all_label, data_NEPALISAT_all_label, data_RAAVANA_all_label,
                            data_UGUISU_all_label], axis=0, ignore_index=True)
"""# 33010*18"""

# train
train_data = data_all.iloc[:int(len(data_all) * 0.65), :]
# train_all_label = data_all_label.iloc[:int(len(data_all_label) * 0.65), :]
"""# 21456*18"""

# test
test_data = data_all.iloc[int(len(data_all) * 0.65):, :]
test_all_label = data_all_label.iloc[int(len(data_all_label) * 0.65):, :]
"""# 11554*18"""

# label
test_label = np.any(test_all_label.values == 1, axis=1)
test_label = pd.DataFrame(test_label, columns=['attack'])
"""# 11554*1"""

# plot
channel_plot('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/train.pdf',
             train_data, channel_list, data_all_label=None)
channel_plot('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/test.pdf',
             test_data, channel_list, data_all_label=test_all_label)

# csv
train_data.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_train.csv', index=False, mode='w',
                  header=True)
train_data.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_train.pkl')

test_data.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test.csv', index=False, mode='w',
                 header=True)
test_data.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test.pkl')

test_all_label.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test_all_label.csv', index=False,
                      mode='w', header=True)
test_all_label.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test_all_label.pkl')

test_label.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test_label.csv', index=False, mode='w',
                  header=True)
test_label.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part/BIRDS_test_label.pkl')

print('finish')





# In[02]: only V and I


# plt.style.reload_library()
plt.style.use('science')
# plt.style.use(['science', 'ieee'])


def channel_plot(plot_dirname_path,
                 data,
                 channel_list=None,
                 data_all_label=None):
    pdf = PdfPages(plot_dirname_path)
    plt.rcParams['figure.figsize'] = data.shape[0] * 6/15000, 1
    for dim in range(data.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(data.values[:, dim], linewidth=0.2, color='k')
        if data_all_label is not None:
            if np.any(data_all_label.values[:, dim]):
                ax_2 = ax.twinx()
                ax_2.fill_between(np.arange(data.shape[0]), data_all_label.values[:, dim], color='red', alpha=0.3)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.set_ylabel(str(channel_list[dim]))
        # fig.tight_layout()
        # fig.tight_layout()是调整整体空白，使得各子图之间的空白更合理
        pdf.savefig(fig)
        plt.close()
    pdf.close()


channel_list = [#'Tpy (°C)', 'Tpx (°C)', 'Tmz (°C)', 'Tmx (°C)', 'Tpz (°C)',
                'Vpy (mV)', 'Vpx (mV)', 'Vmz (mV)', 'Vmx (mV)', 'Vpz (mV)',
                'Ipy (mA)', 'Ipx (mA)', 'Imz (mA)', 'Imx (mA)', 'Ipz (mA)',
                #'Vbat (V)', 'Ibatt(mA)', 'Tbatt (℃)'
]


data_TSURU = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/TSURU.xlsx',
                            sheet_name=None)
data_TSURU.pop('Test1 w batt',None)
data_TSURU.pop('Test2 wo batt',None)
data_TSURU = pd.concat(data_TSURU.values(), ignore_index=True)
data_TSURU = data_TSURU[channel_list]
"""# 24541*18"""

data_NEPALISAT = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/NEPALISAT.xlsx',
                               sheet_name=None)
data_NEPALISAT = pd.concat(data_NEPALISAT.values(), ignore_index=True)
data_NEPALISAT = data_NEPALISAT[channel_list]
"""# 3040*10"""

data_RAAVANA = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/RAAVANA.xlsx',
                             sheet_name=None)
data_RAAVANA = pd.concat(data_RAAVANA.values(), ignore_index=True)
data_RAAVANA = data_RAAVANA[channel_list]
"""# 2159*10"""

data_UGUISU = pd.read_excel('/home/data/DATA/BIRDS_1U_CubeSat/original_data/UGUISU.xlsx',
                            sheet_name=None)
data_UGUISU = pd.concat(data_UGUISU.values(), ignore_index=True)
data_UGUISU = data_UGUISU[channel_list]
"""# 3270*10"""

# label:
# The first type of anomaly comes from RAAVANA
# detail: https://www.mdpi.com/2076-3417/12/17/8634
data_RAAVANA_all_label = data_RAAVANA.copy()
data_RAAVANA_all_label.loc[:, :] = 0
data_RAAVANA_all_label.loc[data_RAAVANA['Vpy (mV)'] < 1200, ['Vpy (mV)', 'Ipy (mA)']] = 1


# The second type of anomaly comes UGUISU
# detail: https://www.mdpi.com/2076-3417/12/17/8634
data_UGUISU_all_label = data_UGUISU.copy()
data_UGUISU_all_label.loc[:, :] = 0
data_UGUISU_all_label.loc[((data_UGUISU['Vpy (mV)'] >= 2000) &
                           (data_UGUISU['Vpy (mV)'] <= 3000) &
                           (data_UGUISU['Ipy (mA)'] < 50)),
                          ['Vpy (mV)', 'Ipy (mA)']] = 1

# TSURU
data_TSURU_all_label = data_TSURU.copy()
data_TSURU_all_label.loc[:, :] = 0

# NEPALISAT
data_NEPALISAT_all_label = data_NEPALISAT.copy()
data_NEPALISAT_all_label.loc[:, :] = 0


# split
data_all = pd.concat([data_TSURU, data_NEPALISAT, data_RAAVANA, data_UGUISU], axis=0, ignore_index=True)
data_all_label = pd.concat([data_TSURU_all_label, data_NEPALISAT_all_label, data_RAAVANA_all_label,
                            data_UGUISU_all_label], axis=0, ignore_index=True)
"""# 33010*18"""

# train
train_data = data_all.iloc[:int(len(data_all) * 0.65), :]
# train_all_label = data_all_label.iloc[:int(len(data_all_label) * 0.65), :]
"""# 21456*18"""

# test
test_data = data_all.iloc[int(len(data_all) * 0.65):, :]
test_all_label = data_all_label.iloc[int(len(data_all_label) * 0.65):, :]
"""# 11554*18"""

# label
test_label = np.any(test_all_label.values == 1, axis=1)
test_label = pd.DataFrame(test_label, columns=['attack'])
"""# 11554*1"""


# plot
channel_plot('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/train.pdf',
             train_data, channel_list, data_all_label=None)
channel_plot('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/test.pdf',
             test_data, channel_list, data_all_label=test_all_label)

# csv
train_data.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_train.csv', index=False, mode='w',
                  header=True)
train_data.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_train.pkl')

test_data.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test.csv', index=False, mode='w',
                 header=True)
test_data.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test.pkl')

test_all_label.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test_all_label.csv', index=False,
                      mode='w', header=True)
test_all_label.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test_all_label.pkl')

test_label.to_csv('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test_label.csv', index=False, mode='w',
                  header=True)
test_label.to_pickle('/home/data/DATA/BIRDS_1U_CubeSat/BIRDS_6535part_10sensor/BIRDS_10sensor_test_label.pkl')

print("finish")


