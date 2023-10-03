# This script runs the AMT experiements
#!/bin/bash
cd "$(/gscratch/xlab/jrfish/authorship_obfuscation_decoding_algorithm "$0")"

# Experiemental Parameters
DATASET=blog
NUM_BEAMS=5
NUM_RETURN_SEQUENCES=5


for NUMBER_AUTHOR in 5 10
do
    # Step 0: Process Data
    echo Step 0: Process Data
    python process_raw_data.py --dataset $DATASET --num_authors $NUMBER_AUTHOR 

    # Step 1: Extract Keywords
    echo Step 1: Extract Keywords
    python main_keyword_extraction.py --dataset $DATASET --num_authors $NUMBER_AUTHOR 
    
    # Step 2: Over-Generation
    echo Step 2: Over-Generation
    python main_generation.py --dataset $DATASET --num_authors $NUMBER_AUTHOR --num_beams $NUM_BEAMS --num_return_sequences $NUM_RETURN_SEQUENCES 

    # Step 3: Filtering
    echo Step 3: Filtering
    python main_filter.py --dataset $DATASET --num_authors $NUMBER_AUTHOR --filter_content_words --filter_rake 
    
    # Step 4: Evaluation
    echo Step 4: Evaluation
    python main_evaluation.py --dataset $DATASET --num_authors $NUMBER_AUTHOR 
done