#!/bin/bash

output_dir=/mnt/jailbreak-defense/exp/johnh/almj/241218_anthropic_batch

echo python3 -m examples.run_anthropic_batch --output_dir $output_dir --limit_dataset 200 --anthropic_num_threads 20 --num_repeats 2 --batch True --anthropic_tag "ANTHROPIC_API_KEY_BATCH"

echo python3 -m examples.run_anthropic_batch --output_dir $output_dir --limit_dataset 200 --anthropic_num_threads 20 --num_repeats 2 --batch False --anthropic_tag "ANTHROPIC_API_KEY"
