#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#define INFINITY INT_MAX

// Cross-platform timing
#ifdef _WIN32
#include <windows.h>
long long get_nanoseconds() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000LL) / freq.QuadPart;
}
#else
#include <time.h>
long long get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
#endif

// Graph structure using adjacency matrix
typedef struct Graph {
    int V;             // Number of vertices
    int E;             // Number of edges
    int** adjMatrix;   // Adjacency matrix
} Graph;

// ------------------------ Graph Creation -------------------------- //
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = 0;
    
    graph->adjMatrix = (int**)malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        graph->adjMatrix[i] = (int*)calloc(V, sizeof(int)); // Initialize with 0
    }
    return graph;
}

void addEdge(Graph* graph, int u, int v, int w) {
    if (graph->adjMatrix[u][v] == 0) graph->E++;
    graph->adjMatrix[u][v] = w;
}

Graph* readGraph(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n%d\n", &E, &V);
    printf("Graph: %d edges, %d vertices\n", E, V);

    Graph* graph = createGraph(V);
    char line[2048];
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);
        while ((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            addEdge(graph, u, v, w);
        }
    }
    fclose(file);
    return graph;
}

// ------------------------ Algorithm Functions -------------------------- //
void initialize_arrays(int *d, int *p, int V, int source) {
    for (int i = 0; i < V; i++) {
        d[i] = INFINITY;
        p[i] = -1;
    }
    d[source] = 0;  // Distance to source is 0
}

void relax_edges(Graph *g, int *d, int *p) {
    for (int u = 0; u < g->V; u++) {
        for (int v = 0; v < g->V; v++) {
            int w = g->adjMatrix[u][v];
            if (w != 0 && d[u] != INFINITY && d[v] > d[u] + w) {
                d[v] = d[u] + w;
                p[v] = u;
            }
        }
    }
}

bool detect_negative_cycles(Graph *g, const int *d) {
    for (int u = 0; u < g->V; u++) {
        for (int v = 0; v < g->V; v++) {
            int w = g->adjMatrix[u][v];
            if (w != 0 && d[u] != INFINITY && d[v] > d[u] + w) {
                return true;  // Negative cycle detected
            }
        }
    }
    return false;
}

void print_results(const int *d, const int *p, int V) {
    printf("Distance array: ");
    for (int i = 0; i < V; i++) {
        printf(d[i] == INFINITY ? "INF " : "%d ", d[i]);
    }
    printf("\nPredecessor array: ");
    for (int i = 0; i < V; i++) printf("%d ", p[i]);
    printf("\n");
}

// ------------------------ Core Algorithm -------------------------- //
void bellmanford(Graph *g, int source, bool debug) {
    int V = g->V;
    int *d = (int*)malloc(V * sizeof(int));
    int *p = (int*)malloc(V * sizeof(int));

    // Step 1: Initialize
    initialize_arrays(d, p, V, source);

    // Step 2: Relax edges V-1 times
    for (int i = 1; i <= V - 1; i++) {
        relax_edges(g, d, p);
    }

    // Step 3: Check for negative cycles
    if (detect_negative_cycles(g, d)) {
        printf("Negative weight cycle detected!\n");
    } else if (debug) {
        print_results(d, p, V);
    }

    free(d);
    free(p);
}

// ------------------------ Main -------------------------- //
int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <graph_file_number> [debug]\n", argv[0]);
        fprintf(stderr, "  debug: 0 (silent) or 1 (verbose, default)\n");
        return 1;
    }

    char filename[50];
	int source = 0;
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

	// Set debug mode (default to true if not specified)
    bool debug = true;
    if (argc == 3) {
        debug = (atoi(argv[2]) != 0);
    }

	// ------------------------ Core Algorithm -------------------------- //

    int V = g->V;
    int *d = (int*)malloc(V * sizeof(int));
    int *p = (int*)malloc(V * sizeof(int));
	
	// Time only the algorithm (excludes graph loading)
	long long tstart = get_nanoseconds();
    // Step 1: Initialize
    initialize_arrays(d, p, V, source);

    // Step 2: Relax edges V-1 times
    for (int i = 1; i <= V - 1; i++) {
        relax_edges(g, d, p);
    }

    // Step 3: Check for negative cycles
    if (detect_negative_cycles(g, d)) {
        printf("Negative weight cycle detected!\n");
    } 
	
	long long tstop = get_nanoseconds();

	if (debug) {
        print_results(d, p, V);
    }

    free(d);
    free(p);

    printf("Algorithm time: %.6f milliseconds\n", (double)(tstop - tstart) / 1e6);

    // Cleanup
    for (int i = 0; i < g->V; i++) free(g->adjMatrix[i]);
    free(g->adjMatrix);
    free(g);
    return 0;
}