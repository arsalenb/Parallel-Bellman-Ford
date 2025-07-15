#!/bin/bash

# Configuration
EXECUTABLE="./cuda_naive"
START=100
END=5000
STEP=100
OUTPUT_FILE="cuda__naive_results.csv"
GRAPH_DIR="graphs"

echo "Vertices,Time (ms)" > $OUTPUT_FILE

for ((vertices=$START; vertices<=$END; vertices+=$STEP)); do
    graph_file="${GRAPH_DIR}/graph_${vertices}.txt"
    echo "Processing $graph_file..."
    
    output=$($EXECUTABLE $vertices 0 2>&1)
    time_ms=$(echo "$output" | grep -oP "Total algorithm time: \K[0-9.]+")
    
    if [ -z "$time_ms" ]; then
        echo "ERROR: Failed to get time for $vertices vertices"
        time_ms="FAILED"
    fi
    
    echo "$vertices,$time_ms" >> $OUTPUT_FILE
    echo "  -> $vertices vertices: $time_ms ms"
done

echo "All tests completed! Results saved to $OUTPUT_FILE"
