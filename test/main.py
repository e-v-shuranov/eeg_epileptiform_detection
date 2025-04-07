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


# reference_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_test/"
# hypothesis_folder = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result_test/"
# output_file = "/home/eshuranov/projects/test/BIDS_Siena_test.json"
#
#
# result = evaluate_dataset(reference_folder, hypothesis_folder, output_file)
# print(result)

# reference_folder = "/media/public/Datasets/MATBII/bids_true_no_sz_labels/"
# hypothesis_folder = "/media/public/Datasets/MATBII/bids_v8_results/"
# output_file = "/home/eshuranov/projects/test/MATBII_v8_best.json"


# input = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena"
# result = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_v8_best"
# output_file = "/home/eshuranov/projects/test/result_scores/BIDS_Siena_v8_best_xgb_model_2025_4_level_GPU.json"

# result = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_v8_best_xgb_model_2025_4_level_GPU"

input = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess"
result = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_v8_best_recheck"
output_file = "/home/eshuranov/projects/test/result_scores/tuh_train_preprocess_v8_best_recheck.json"
result = evaluate_dataset(input, result, output_file)

result = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_v8_best_xgb_model_2025_4_level_GPU"
output_file = "/home/eshuranov/projects/test/result_scores/tuh_train_preprocess_v8_best_xgb_model_2025_4_level_GPU.json"

result = evaluate_dataset(input, result, output_file)
print(result)