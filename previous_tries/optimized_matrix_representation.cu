#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define INFINITY 99999

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

// Graph structure using adjacency matrix
typedef struct Graph {
    int V;             // Number of vertices
    int E;             // Number of edges
    int** adjMatrix;   // Adjacency matrix
} Graph;

// Create a graph with V vertices
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = 0;
    
    // Allocate memory for adjacency matrix
    graph->adjMatrix = (int**)malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        graph->adjMatrix[i] = (int*)malloc(V * sizeof(int));
        for (int j = 0; j < V; j++) {
            graph->adjMatrix[i][j] = 0; // Initialize with 0 (no edge)
        }
    }
    
    return graph;
}

// Add an edge to the graph
void addEdge(Graph* graph, int u, int v, int w) {
    if (graph->adjMatrix[u][v] == 0) {
        graph->E++;
    }
    graph->adjMatrix[u][v] = w;
}

// Read graph from file and build adjacency matrix
Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    printf("Edges: %d, Vertices: %d\n", E, V);

    Graph* graph = createGraph(V);
    char line[2048];
    char* token;

    while(fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, ":");
        int u = atoi(token);

        while((token = strtok(NULL, ";")) != NULL) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            addEdge(graph, u, v, w);
        }
    }
    fclose(file);
    return graph;
}

// Free graph memory
void freeGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) {
        free(g->adjMatrix[i]);
    }
    free(g->adjMatrix);
    free(g);
}

// CUDA Kernel for initialization
__global__ void cudaInitialization(int* dist, int src_idx, int* mask, int* mask1, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        dist[idx] = INFINITY;
        mask[idx] = 0;
        mask1[idx] = 0;
        if (idx == src_idx) {
            dist[idx] = 0;
            mask[idx] = 1;
        }
    }
}

// CUDA Kernel for relaxation
__global__ void cudaRelaxation(int* dist, int* matrix, int* mask, int* mask1, int* flag, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < V && j < V) {
        if (mask[i] == 1) {
            int weight = matrix[i * V + j];
            if (weight != 0) {  // If there's an edge
                int new_dist = dist[i] + weight;
                if (new_dist < dist[j]) {
                    atomicMin(&dist[j], new_dist);
                    mask1[j] = 1;
                    *flag = 1;
                }
            }
        }
    }
}

// CUDA Kernel for swapping masks
__global__ void cudaSwap(int* mask, int* mask1, int* flag, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        mask[idx] = mask1[idx];
        mask1[idx] = 0;
    }
    if (idx == 0) {
        *flag = 0;
    }
}

// CUDA Bellman-Ford implementation
void cudaBellmanFord(Graph* g, int source) {
    int V = g->V;
    int* dist = (int*)malloc(V * sizeof(int));
    int* p = (int*)malloc(V * sizeof(int));  // Predecessor array (not used in CUDA version)
    
    // Device pointers
    int *d_dist, *d_matrix, *d_mask, *d_mask1, *d_flag;
    
    // Allocate device memory
    cudaMalloc((void**)&d_dist, V * sizeof(int));
    cudaMalloc((void**)&d_matrix, V * V * sizeof(int));
    cudaMalloc((void**)&d_mask, V * sizeof(int));
    cudaMalloc((void**)&d_mask1, V * sizeof(int));
    cudaMalloc((void**)&d_flag, sizeof(int));
    
    // Flatten adjacency matrix
    int* flatMatrix = (int*)malloc(V * V * sizeof(int));
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            flatMatrix[i * V + j] = g->adjMatrix[i][j];
        }
    }
    
    // Copy matrix to device
    cudaMemcpy(d_matrix, flatMatrix, V * V * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize kernel configuration
    dim3 blockSize(256);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x);
    
    // Initialize on device
    cudaInitialization<<<gridSize, blockSize>>>(d_dist, source, d_mask, d_mask1, V);
    cudaDeviceSynchronize();
    
    // Relaxation kernel configuration
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((V + blockSize2D.x - 1) / blockSize2D.x, 
                    (V + blockSize2D.y - 1) / blockSize2D.y);
    
    int flag = 1;
    int iterations = 0;
    
    for (int i = 1; i <= V - 1 && flag; i++) {
        flag = 0;
        cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
        
        cudaRelaxation<<<gridSize2D, blockSize2D>>>(d_dist, d_matrix, d_mask, d_mask1, d_flag, V);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaSwap<<<gridSize, blockSize>>>(d_mask, d_mask1, d_flag, V);
        cudaDeviceSynchronize();
        
        iterations++;
    }
    
    printf("Completed in %d iterations\n", iterations);
    
    // Check for negative cycles
    flag = 0;
    cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
    cudaRelaxation<<<gridSize2D, blockSize2D>>>(d_dist, d_matrix, d_mask, d_mask1, d_flag, V);
    cudaDeviceSynchronize();
    cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (flag) {
        printf("Negative weight cycle detected!\n");
    } else {
        // Copy results back to host
        cudaMemcpy(dist, d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Distance array: ");
        for (int i = 0; i < V; i++) {
            if (dist[i] == INFINITY) {
                printf("INF ");
            } else {
                printf("%d ", dist[i]);
            }
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_dist);
    cudaFree(d_matrix);
    cudaFree(d_mask);
    cudaFree(d_mask1);
    cudaFree(d_flag);
    free(flatMatrix);
    free(dist);
    free(p);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
        return 1;
    }
    
    char filename[50];
    int arg = atoi(argv[1]);
    sprintf(filename, "graphs/graph_%d.txt", arg);
    
    Graph* g = readGraph(filename);
    if (!g) {
        return 1;
    }
	
	double tstart, tstop;

    tstart = gettime();
    cudaBellmanFord(g, 0);
    tstop = gettime();
	
    printf("Total algorithm time: %.9f seconds\n", tstop - tstart);

    freeGraph(g);
    return 0;
}