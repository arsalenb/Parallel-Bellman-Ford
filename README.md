# 588II – Computer Architecture: Parallel Bellman-Ford Implementations

This project investigates and benchmarks multiple parallel implementations of the **Bellman-Ford algorithm**, focusing on performance trade-offs across different parallelization strategies. The solution was developed in **C**, leveraging **Pthreads** for CPU multithreading and **CUDA** for GPU acceleration. It was completed as the final project for the Computer Architecture course.

## Project Structure

The project consists of several **executables** and **scripts** used for implementation, testing, and performance evaluation.

### Executables
- **Sequential version** – Establishes a performance baseline on a single thread/processor. Includes two graph representation variants (adjacency matrix and edge list) to analyze memory footprint.  
- **CPU parallelized version** – Implements multithreading on the CPU using Pthreads.  
- **Optimized CPU parallelized version** – Algorithmically optimized to support early stopping, improving performance over the basic CPU parallel version.  
- **Naive CUDA version** – GPU-parallelized implementation of Bellman-Ford.  
- **Optimized CUDA version** – Algorithmically equivalent to the naive GPU version, with optimizations for improved performance.

### Scripts
- **Graph generator script** – Creates large graphs to build datasets for testing and benchmarking.  
- **Execution and reporting scripts (Bash and PowerShell)** – Automate running executables and recording results in CSV files.  
- **Thread variation script** – Adjusts the number of threads in CPU executables and calculates average results over multiple trials.

---

## Compilation & Execution

### CPU Executables (C / Pthreads)

Compile using `g++`:

```bash
# Sequential version
g++ -O2 -std=c++17 -o sequential_edgelist sequential_edgelist.cpp
g++ -O2 -std=c++17 -o sequential_adjacency sequential_adjacency.cpp

# CPU parallelized version
g++ -O2 -std=c++17 -pthread -o cpu_parallel cpu_parallel.cpp

# Optimized CPU parallelized version
g++ -O2 -std=c++17 -pthread -o cpu_parallel_optimized cpu_parallel_optimized.cpp
```

Run the executables:

```bash
Syntax: ./executable <num_vertices> <silent_flag>

<num_vertices> – Number of vertices in the graph
<silent_flag> – 0 for silent mode (suppress extra output)
```

### GPU Executables (CUDA)

Compile using `nvcc`:

```bash
# Naive CUDA version
nvcc -O2 -arch=sm_70 -o naive_cuda naive_cuda.cu

# Optimized CUDA version
nvcc -O2 -arch=sm_70 -o cuda_optimized cuda_optimized.cu
```

Run the executables:

```bash
./naive_cuda 1000 0
./cuda_optimized 2000 0

<num_vertices> – Number of vertices in the graph
<silent_flag> – 0 for silent mode
```
