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

__global__ void relax_kernel(Edge* edges, int* d, int* p, int* d_temp, int E, int V) {
    extern __shared__ int shared_d[];  // Dynamic shared memory

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tx < V) {
        shared_d[tx] = d[tx];
    }
    __syncthreads();

    if (idx >= E) return;

    Edge e = edges[idx];
    int u = e.u;
    int v = e.v;
    int w = e.w;

    if (u < V && shared_d[u] != INFINITY) {  // Bounds check
        int new_dist = shared_d[u] + w;
        int old_dist = atomicMin(&d_temp[v], new_dist);
        if (old_dist > new_dist) {
            p[v] = u;
        }
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
    Edge* d_edges;
    int *d_d, *d_p, *d_temp;
    int *h_d, *h_p;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_edges, g->E * sizeof(Edge)));
    CUDA_CHECK(cudaMemcpy(d_edges, g->edge, g->E * sizeof(Edge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_d, g->V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p, g->V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, g->V * sizeof(int)));  // Temporary array

    // Initialize device arrays
    int threadsPerBlock = 256;
    int blocks = (g->V + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock>>>(d_d, d_p, g->V, source);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run relaxation for V-1 iterations
    int relaxBlocks = (g->E + 255) / 256;
    for (int i = 0; i < g->V - 1; i++) {
        // Copy d_d to d_temp before the kernel runs
        CUDA_CHECK(cudaMemcpy(d_temp, d_d, g->V * sizeof(int), cudaMemcpyDeviceToDevice));
		size_t shared_mem_size = g->V * sizeof(int);
		relax_kernel<<<relaxBlocks, 256, shared_mem_size>>>(d_edges, d_d, d_p, d_temp, g->E, g->V);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap d_d and d_temp
        int* temp = d_d;
        d_d = d_temp;
        d_temp = temp;
    }

    // Allocate host memory for results and copy
    h_d = (int*)malloc(g->V * sizeof(int));
    h_p = (int*)malloc(g->V * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_d, d_d, g->V * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p, d_p, g->V * sizeof(int), cudaMemcpyDeviceToHost));

    // Check for negative cycles and print results
    int hasNegative = 0;
    for (int i = 0; i < g->E; i++) {
        Edge e = g->edge[i];
        if (h_d[e.u] != INFINITY && h_d[e.v] > h_d[e.u] + e.w) {
            hasNegative = 1;
            break;
        }
    }

    if (hasNegative) {
        printf("Negative weight cycle detected!\n");
    } else {
        printf("Distance array: ");
        display(h_d, g->V);
        printf("Predecessor array: ");
        display(h_p, g->V);
    }

    // Cleanup
    free(h_d);
    free(h_p);
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_temp));
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