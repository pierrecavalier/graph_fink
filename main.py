from analysis import *
from vizualisation import *
import datetime
import os

liste = [
    'EB*', 'Mira', 'SN candidate', 'QSO', 'BLLac', 'Blazar',
    'Ambiguous', 'RRLyr', 'YSO', 'LPV*', 'AGN', 'Seyfert_1', 'AGN_Candidate', 'TTau*', 'Kilonova candidate'
]

df = load_subset_data(liste)
dic = {'variable_star_class': ['EB*', 'Mira', 'RRLyr', 'YSO', 'LPV*', "TTau*"],
       'AGN_class': ['QSO', 'BLLac', 'Blazar',
                     'AGN', 'Seyfert_1', 'AGN_Candidate']
       }
df['finkclass'] = df['finkclass'].apply(define_meta_class, dic=dic)

colors = color_map(df['finkclass'].value_counts().index.tolist())

cols = [
    'rf_kn_vs_nonkn',
    'rf_snia_vs_nonia',
    'snn_sn_vs_all',
    'snn_snia_vs_nonia'
]


cols_in_candidate = [
    'jdstarthist',
    'magpsf',
    'sigmapsf',
    'fid',
    'magnr',
    'sigmagnr',
    'isdiffpos',
    'neargaia',
    'sgscore1',
    'classtar'
]

df_filtered = feature_choice(
    df, cols + ['lc_features_g', 'lc_features_r', ['candidate', cols_in_candidate]])

df_filtered = normalize_data(df_filtered)

df_filt_selected = keep_important_variables(df_filtered)

label = df['finkclass']

# take a random sample of 100 elements in df_filt_selected and label
df_filt_selected_sample = df_filt_selected.sample(n=300)
label_sample = label[df_filt_selected_sample.index]


# create an other sample with 100 different elements of df_filt_selected and label
df_filt_selected_sample2 = df_filt_selected.drop(df_filt_selected_sample.index)
df_filt_selected_sample2 = df_filt_selected_sample2.sample(n=100)
label_sample2 = label[df_filt_selected_sample2.index]


X_train, y_train = create_pairs(df_filt_selected_sample, label_sample)
X_test, y_test = create_pairs(df_filt_selected_sample2, label_sample2)

trainloader = DataLoader(dataset(X_train, y_train),
                         batch_size=16, shuffle=False)

testloader = DataLoader(dataset(X_test, y_test), batch_size=64, shuffle=False)

net = Net(len(df_filt_selected.columns.tolist()))

learning_rate = 0.001
epochs = 500
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

errors = global_loop(trainloader, testloader, net,
                     loss_fn, optimizer, epochs=epochs)

folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
path = os.path.join('results/models/', folder)

os.mkdir(path)

torch.save(net.state_dict(), 'results/models/' + folder + '/model.pt')
np.savetxt('results/models/' + folder + '/errors.csv', errors, delimiter=',')
save_error(errors, 'results/models/' + folder)

print("Script ended")
