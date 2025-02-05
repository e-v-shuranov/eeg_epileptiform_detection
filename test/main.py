from szcore_evaluation.evaluate import evaluate_dataset

# Указываем пути к папкам с аннотациями и выходной файл
#reference_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena/"
#hypothesis_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result/"

#reference_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT/"
#hypothesis_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT_result/"
#output_file = "/home/eshuranov/projects/test/res_MIT.json"

#reference_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess/"
#hypothesis_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_result_baseline/"
#output_file = "/home/eshuranov/projects/test/res_TUSZ.json"


reference_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_test/"
hypothesis_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result_test/"
output_file = "/home/eshuranov/projects/test/BIDS_Siena_test.json"


result = evaluate_dataset(reference_folder, hypothesis_folder, output_file)
print(result)