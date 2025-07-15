# Configuration
$executable = "./parallel_otimized_edgelist_threads_varied.exe"
$graph_number = 1000  # Default graph to test
$output_file = "thread_scaling_results.csv"
$max_threads = 50     # Maximum number of threads to test
$trials = 3           # Number of trials per thread count

# Write CSV header
"Threads,Time (ms)" | Out-File -FilePath $output_file

Write-Host "Testing thread scaling from 1 to $max_threads threads (graph $graph_number)"
Write-Host "Running $trials trials per thread count..."
Write-Host "Results will be saved to $output_file"

# Test different thread counts
for ($threads = 1; $threads -le $max_threads; $threads++) {
    $total_time = 0
    
    Write-Host "Testing with $threads threads..."
    
    # Run multiple trials
    for ($trial = 1; $trial -le $trials; $trial++) {
        # Run program with silent mode (0) and capture output
        $result = & $executable $graph_number $threads 0
        
        # Extract execution time (matches your printf format)
        $time_line = $result | Select-String -Pattern "Algorithm time: ([\d.]+) milliseconds"
        if ($time_line) {
            $time_ms = [double]$time_line.Matches.Groups[1].Value
            $total_time += $time_ms
            Write-Host "  Trial $trial : $time_ms ms"
        } else {
            Write-Host "  ERROR: Could not parse time from output"
            $time_ms = 0
        }
    }
    
    # Calculate average time
    $avg_time = $total_time / $trials
    
    # Save results
    "$threads,$avg_time" | Out-File -FilePath $output_file -Append
    Write-Host "  Average time: $avg_time ms`n"
}

Write-Host "Testing completed! Results saved to $output_file"