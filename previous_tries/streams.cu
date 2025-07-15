#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define INFINITY INT_MAX

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifdef _WIN32
#include <windows.h>
long long gettime(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000LL) / freq.QuadPart;
}
#else
#include <time.h>
double gettime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}
#endif

typedef struct Edge {
    int u;
    int v;
    int w;
} Edge;

typedef struct Graph {
    int V;
    int E;
    struct Edge *edge;
} Graph;

// CUDA kernels
__global__ void init_kernel(int* d, int* p, int V, int source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        d[idx] = (idx == source) ? 0 : INFINITY;
        p[idx] = -1;
    }
}

__global__ void relax_kernel(Edge* edges, int* d, int* p, int* d_temp, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    Edge e = edges[idx];
    int u = e.u;
    int v = e.v;
    int w = e.w;

    if (d[u] != INFINITY) {
        int new_dist = d[u] + w;
        int old_dist = atomicMin(&d_temp[v], new_dist);
        if (old_dist > new_dist) p[v] = u;
    }
}


Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(graph->E * sizeof(Edge));
    return graph;
}

Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    Graph* graph = createGraph(V, E);
    int i = 0;
    char line[2048];

    while(fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);

        while((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            if (i > 0 && u == graph->edge[i-1].u && v == graph->edge[i-1].v)
                continue;
            graph->edge[i].u = u;
            graph->edge[i].v = v;
            graph->edge[i].w = w;
            i++;
        }
    }
    fclose(file);
    return graph;
}

void display(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == INFINITY) {
            printf("INF ");
        } else {
            printf("%d ", arr[i]);
        }
    }
    printf("\n");
}

void cuda_bellmanford(Graph* g, int source) {
    // Allocate device memory first
    Edge* d_edges;
    int *d_d, *d_p, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_edges, g->E * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_d, g->V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p, g->V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, g->V * sizeof(int)));

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Split edge list into two chunks
    int E_half = g->E / 2;
    
    // Async memory copies for each chunk
    CUDA_CHECK(cudaMemcpyAsync(d_edges, g->edge, E_half*sizeof(Edge), 
                             cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_edges+E_half, g->edge+E_half, 
                             (g->E-E_half)*sizeof(Edge), cudaMemcpyHostToDevice, stream2));

    // Initialize device arrays
    int threadsPerBlock = 512;
    int blocks = (g->V + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_d, d_p, g->V, source);
    
    // Run relaxation for V-1 iterations
    int relaxBlocks1 = (E_half + 511) / 512;
    int relaxBlocks2 = ((g->E - E_half) + 511) / 512;
    
    for (int i = 0; i < g->V - 1; i++) {
        // Copy d_d to d_temp before the kernel runs
        CUDA_CHECK(cudaMemcpyAsync(d_temp, d_d, g->V * sizeof(int), 
                                 cudaMemcpyDeviceToDevice, stream1));
        
        // Launch kernels in separate streams
        relax_kernel<<<relaxBlocks1, 512, 0, stream1>>>(d_edges, d_d, d_p, d_temp, E_half);
        relax_kernel<<<relaxBlocks2, 512, 0, stream2>>>(d_edges+E_half, d_d, d_p, d_temp, g->E-E_half);
        
        // Synchronize both streams
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));

        // Swap d_d and d_temp
        int* temp = d_d;
        d_d = d_temp;
        d_temp = temp;
    }

    // Copy results back to host
    int *h_d = (int*)malloc(g->V * sizeof(int));
    int *h_p = (int*)malloc(g->V * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_d, d_d, g->V * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p, d_p, g->V * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Distance array: ");
    display(h_d, g->V);
    printf("Predecessor array: ");
    display(h_p, g->V);

    // Cleanup
    free(h_d);
    free(h_p);
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
        return 1;
    }

    char filename[50];
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

    double tstart, tstop;

    tstart = gettime();
    cuda_bellmanford(g, 0);  // 0 is the source vertex
    tstop = gettime();
    printf("Total algorithm time: %.9f seconds\n", tstop - tstart);


    // Cleanup
    free(g->edge);
    free(g);

    return 0;
}