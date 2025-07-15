#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <direct.h>
#include <string.h>

#define SEED 42
#define MAX_WEIGHT 50 // max POSITIVE weight value of an edge
#define MIN_WEIGHT 1  // min weight value of an edge

typedef struct Edge {
    int vertex;
    int weight;
    struct Edge* next;  // Pointer to next edge
} Edge;

typedef struct {
    Edge* edges;  // Linked list of edges
} Vertex;

int main() {
    srand(SEED);

    // Create "graphs" directory if it doesn't exist
    _mkdir("graphs");

    // Generate graphs from 100 to 5000 vertices in steps of 100
    int start = 100;
    int end = 5000;
    int step = 100;
    int num_counts = (end - start) / step + 1;

    for (int i = 0; i < num_counts; i++) {
        int numberOfVertices = start + i * step;
        Vertex* graph = (Vertex*)malloc(sizeof(Vertex) * numberOfVertices);
        int totEdges = 0;

        // Initialize each vertex (no edges yet)
        for (int i = 0; i < numberOfVertices; i++) {
            graph[i].edges = NULL;
        }

        // Generate edges for each vertex
        for (int vertexCounter = 0; vertexCounter < numberOfVertices; vertexCounter++) {
            int* connected = calloc(numberOfVertices, sizeof(int));

            // Each vertex connects to approximately 50% of other vertices
            for (int edgeCounter = 0; edgeCounter < numberOfVertices / 2; edgeCounter++) {
                if (rand() % 2 == 1) {  // 50% chance to create an edge
                    int linkedVertex;
                    do {
                        linkedVertex = rand() % numberOfVertices;
                    } while (linkedVertex == vertexCounter || connected[linkedVertex]);

                    // Create a new edge and add it to the linked list
                    Edge* newEdge = (Edge*)malloc(sizeof(Edge));
                    newEdge->vertex = linkedVertex;
                    newEdge->weight = (rand() % MAX_WEIGHT) + MIN_WEIGHT;
                    newEdge->next = graph[vertexCounter].edges;
                    graph[vertexCounter].edges = newEdge;

                    connected[linkedVertex] = 1;
                    totEdges++;
                }
            }
            free(connected);
        }

        // Write the graph to file
        char filename[50];
        sprintf(filename, "graphs/graph_%d.txt", numberOfVertices);
        FILE* file = fopen(filename, "w");
        if (file == NULL) {
            printf("Could not open file for writing\n");
            exit(1);
        }

        // Write header (total edges and vertices)
        fprintf(file, "%d\n%d\n", totEdges, numberOfVertices);

        // Write edges for each vertex
        for (int i = 0; i < numberOfVertices; i++) {
            fprintf(file, "%d:", i);
            Edge* current = graph[i].edges;
            while (current != NULL) {
                fprintf(file, "%d,%d", current->vertex, current->weight);
                current = current->next;
                if (current != NULL) {
                    fprintf(file, ";");
                }
            }
            fprintf(file, "\n");
        }

        fclose(file);
        printf("Generated graph with %d vertices and %d edges\n", numberOfVertices, totEdges);

        // Free memory
        for (int i = 0; i < numberOfVertices; i++) {
            Edge* current = graph[i].edges;
            while (current != NULL) {
                Edge* temp = current;
                current = current->next;
                free(temp);
            }
        }
        free(graph);
    }

    return 0;
}