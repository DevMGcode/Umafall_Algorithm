def run_extraction():
    import scripts.one_extract_data_wrist_filter as extract_wrist_data_script
    extract_wrist_data_script.process_all_files(
        "UMAFall_Dataset.zip",
        "data/wrist_filtered/wrist_data_corrected.csv"
    )

def run_preprocessing():
    import scripts.two_preprocess_extract_features as preprocess_script
    preprocess_script.preprocess_data(
        "data/wrist_filtered/wrist_data_corrected.csv",
        "data/wrist_filtered/preprocessed/wrist_data_preprocessed_segmented.csv",
        "data/wrist_filtered/features/wrist_features.csv",
        "data/wrist_filtered/features/wrist_features_optimized.csv"
    )

def run_eda():
    import scripts.three_eda as eda_script
    eda_script.run_eda()

def run_training():
    import scripts.four_train_model as train_script
    train_script.main()

def run_evaluation():
    import scripts.five_evaluate_model as evaluate_script
    evaluate_script.main()

if __name__ == "__main__":
    run_extraction()
    run_preprocessing()
    run_eda()
    run_training()
    run_evaluation()
    print("Pipeline completado.")