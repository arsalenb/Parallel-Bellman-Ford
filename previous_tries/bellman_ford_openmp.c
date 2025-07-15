#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#define INFINITY 99999

#ifdef _WIN32
#include <windows.h>
double gettime(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / freq.QuadPart;
}
#else
#include <time.h>
double gettime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}
#endif

typedef struct Edge { int u, v, w; } Edge;
typedef struct Graph { int V, E; Edge *edge; } Graph;

Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(E * sizeof(Edge));
    return graph;
}

Graph* readGraph(char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file: %s\n", filename);
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n%d\n", &E, &V);

    Graph* graph = createGraph(V, E);
    char line[2048];
    int i = 0;

    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);

        while ((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            graph->edge[i].u = u;
            graph->edge[i].v = v;
            graph->edge[i].w = w;
            i++;
        }
    }

    fclose(file);
    return graph;
}

// Modified Bellman-Ford to suppress output
void bellmanford(struct Graph *g, int source) {
    int *d = (int*)malloc(g->V * sizeof(int));
    for (int i = 0; i < g->V; i++) d[i] = INFINITY;
    d[source] = 0;

    for (int i = 1; i < g->V; i++) {
        #pragma omp parallel for  // Parallelize edge processing
        for (int j = 0; j < g->E; j++) {
            int u = g->edge[j].u, v = g->edge[j].v, w = g->edge[j].w;
            if (d[u] != INFINITY && d[v] > d[u] + w) {
                #pragma omp critical  // Protect concurrent writes
                if (d[v] > d[u] + w) d[v] = d[u] + w;  // Recheck condition
            }
        }
    }
    free(d);
}

int main() {
	printf("Using %d threads\n", omp_get_max_threads());

    FILE* csv = fopen("results.csv", "w");
    if (!csv) {
        printf("Failed to create CSV file.\n");
        return 1;
    }
    fprintf(csv, "Vertices,Edges,Time(seconds)\n");

    for (int vertices = 10; vertices <= 2500; vertices += 10) {
        char filename[50];
        sprintf(filename, "graphs/graph_%d.txt", vertices);

        Graph* g = readGraph(filename);
        if (!g) continue;

        // Warm-up run (optional)
        bellmanford(g, 0);

        // Timed run
        double tstart = gettime();
        bellmanford(g, 0);
        double tstop = gettime();
        double elapsed = tstop - tstart;

        fprintf(csv, "%d,%d,%.9f\n", vertices, g->E, elapsed);
        printf("Processed %d vertices: %.6f sec\n", vertices, elapsed);

        free(g->edge);
        free(g);
    }

    fclose(csv);
    return 0;
}