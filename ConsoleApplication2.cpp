#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "mpi.h"
using namespace std;
vector<vector<int>> generateConnectedGraph(int N, int minWeight, int maxWeight)
{
    const int INF = 1e9;
    vector<vector<int>> adjMatrix(N, vector<int>(N, INF));
    srand(time(0));
    for (int i = 0; i < N; ++i) adjMatrix[i][i] = 0;
    for (int i = 1; i < N; ++i)
    {
        int weight = minWeight + rand() % (maxWeight - minWeight + 1);
        adjMatrix[i][i - 1] = weight;
        adjMatrix[i - 1][i] = weight;
    }
    int extraEdges = rand() % (N * (N - 1) / 2 - (N - 1));
    for (int i = 1; i <= extraEdges; ++i)
    {
        int u = rand() % N;
        int v = rand() % N;
        if ((u != v) && (abs(u - v) != 1) && (adjMatrix[u][v] == INF))
        {
            int weight = minWeight + rand() % (maxWeight - minWeight + 1);
            adjMatrix[u][v] = adjMatrix[v][u] = weight;
        }
    }
    return adjMatrix;
}
void adjMatrixToFile(const vector<vector<int>>& matrix, const string& filename) 
{
    int N=matrix.size();
    ofstream fout(filename);
    for (int i=0;i<N;++i)
    {
        for (int j = 0; j < N;++j) fout << matrix[i][j] << " ";
        fout << "\n";
    }
    fout.close();
}
void answersToFile(const vector<vector<int>>& matrix, const string& filename)
{
    int N = matrix.size();
    ofstream fout(filename);
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j) fout << "The shortest distance between " << i << " and " << j << " is: " << matrix[i][j] << "\n";
    }
    fout.close();
}
int procOfRow(int N, int size, int row)
{
    int baseRows = N / size;
    int extra = N % size;
    int extraBorder = extra* (baseRows + 1);
    if (row<extraBorder) return row / (baseRows + 1);
    int postBorder = row - extraBorder;
    return min(size-1,extra + (postBorder / baseRows));
}
int main(int argc, char** argv) 
{
    setlocale(LC_CTYPE, "ukr");
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N;
    vector<vector<int>> fullMatrix;
    if (rank == 0) 
    {
        cout << "Enter the number of vertexes: ";
        cin >> N;
        fullMatrix = generateConnectedGraph(N,1,10);
        adjMatrixToFile(fullMatrix, "adjacency_matrix.txt");
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int baseRows = N / size;
    int extra = N % size;
    int procRows = baseRows + (rank < extra ? 1 : 0);
    int startRow = baseRows * rank + min(rank, extra);
    vector<vector<int>> localMatrix(procRows, vector<int>(N));
    vector<int> flatMatrix;
    if (rank == 0) 
    {
        flatMatrix.resize(N * N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j) flatMatrix[i * N + j] = fullMatrix[i][j];                
        }
    }
    vector<int> partSizes(size), offsets(size);
    for (int i = 0; i < size; ++i) 
    {
        int rows = baseRows + (i < extra ? 1 : 0);
        partSizes[i] = rows * N;
        offsets[i] = (baseRows * i + min(i, extra)) * N;
    }
    vector<int> localData(procRows * N);
    MPI_Scatterv(flatMatrix.data(), partSizes.data(), offsets.data(), MPI_INT,
        localData.data(), procRows * N, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < procRows; ++i)
    {
        for (int j = 0; j < N; ++j) localMatrix[i][j] = localData[i * N + j];            
    }
    double prog_start = MPI_Wtime();
    vector<int> kRow(N);
    for (int k = 0; k < N; ++k) 
    {
        if ((startRow <= k) && (k < startRow + procRows)) 
        {
            for (int j = 0; j < N; ++j) kRow[j] = localMatrix[k - startRow][j];               
        }
        MPI_Bcast(kRow.data(), N, MPI_INT, procOfRow(N,size,k), MPI_COMM_WORLD);
        for (int i = 0; i < procRows; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (localMatrix[i][k] != 1e9 && kRow[j] != 1e9 && (localMatrix[i][k] + kRow[j] < localMatrix[i][j]))
                    localMatrix[i][j] = localMatrix[i][k] + kRow[j];
            }
        }        
    }
    double prog_end = MPI_Wtime();
    double prog_time = prog_end - prog_start;
    vector<int> finalMatrix;
    if (rank == 0) finalMatrix.resize(N * N);
    for (int i = 0; i < procRows; ++i)
    {
        for (int j = 0; j < N; ++j) localData[i * N + j] = localMatrix[i][j];            
    }
    MPI_Gatherv(localData.data(), procRows * N, MPI_INT,
        finalMatrix.data(), partSizes.data(), offsets.data(), MPI_INT,
        0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        vector<vector<int>> shortestPaths(N, vector<int>(N));
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j) shortestPaths[i][j] = finalMatrix[i * N + j];                
        }
        answersToFile(shortestPaths, "shortest_paths.txt");
        cout << "Execution time: " << prog_time << " seconds\n";
    }
    MPI_Finalize();
    return 0;
}


