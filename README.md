# eeg_epileptiform_detection
 eeg epileptiform detection 

17_02_2025:
    Solution was sent to Seizure detection challenge (2025)
    https://epilepsybenchmarks.com/challenge/#organizers
    
    
    Finetune: run_class_finetuning_sz_chlng_2025.py
    best checkpoint v8: 
    https://hub.docker.com/r/eegdude/solution2

    for tests 
    https://github.com/e-v-shuranov/szCORE_challenge
    szCORE_challenge\solution1_master\labram_based_solution\src\run_labram_based_test.py
    
    for metric evaluation run:
    from szcore_evaluation.evaluate import evaluate_dataset
    result = evaluate_dataset(reference_folder, hypothesis_folder, output_file)

