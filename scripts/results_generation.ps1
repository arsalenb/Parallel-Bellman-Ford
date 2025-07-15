# Configuration
$executable = ".\sequential_edgelist.exe"  # Your compiled executable
$start = 100
$end = 5000
$step = 100
$outputFile = "sequential_edgelist_results.csv"

# Clear previous results
"Vertices,Time (ms)" | Out-File -FilePath $outputFile

# Process each graph size
for ($vertices = $start; $vertices -le $end; $vertices += $step) {
    Write-Host "Processing graph_$vertices.txt..."
    
    # Run the program and capture output
    $result = & $executable $vertices 0  # '0' for silent mode
    
    # Initialize time variable
    $time_ms = "FAILED"
    
    # Extract time using regex
    foreach ($line in $result) {
        if ($line -match "Algorithm time: ([0-9.]+) milliseconds") {
            $time_ms = $matches[1]
            break
        }
    }
    
    # Save results
    "$vertices,$time_ms" | Out-File -FilePath $outputFile -Append
    Write-Host "  -> $vertices vertices: $time_ms ms"
}

Write-Host "All tests completed! Results saved to $outputFile"