
"Converting sz datasets to bids format"

from epilepsy2bids.bids.tuh.convert2bids import convert

# inputpath = '/media/public/Datasets/TUEV/tuev/edf/train'
# outpath = '/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_test_preprocess'
# inputpath = '/media/public/Datasets/TUEV/tuev/edf/train/aaaaaaar'
# inputpath = '/home/eshuranov/projects/eeg_epileptiform_detection/epilepsy2bids-main/tests/data/tuh/train/aaaaapks/s018_2014/03_tcp_ar_a/aaaaapks_s018_t010.edf'
# outpath = '/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess'

# inputpath = '/media/public/Datasets/TUSZ_v2.0.3/edf'
# outpath = '/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess'

inputpath = '/media/public/Datasets/TUSZ_v2.0.3/edf'
outpath = '/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_qqq'


convert(inputpath, outpath)


