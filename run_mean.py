import pickle

from utils_csc import double_correlation_clustering

# all_atoms_info = pd.read_csv('./all_atoms_info.csv')
atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))

exclude_subs = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

atom_groups, group_summary = double_correlation_clustering(
    atom_df, u_thresh=0.4, v_thresh=0.4, exclude_subs=exclude_subs,
    output_dir=None)
