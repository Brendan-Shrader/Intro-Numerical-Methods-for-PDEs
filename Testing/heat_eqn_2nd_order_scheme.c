#include <stdio.h>
#include <stdlib.h>

// Structures
// ----------
typedef struct MeshStruct {
    double** mesh_arr;
    double a, b, T, BC_a, BC_b;
    int t_res;
    int x_res;
} MeshStruct;

// Function declarations
// ---------------------
MeshStruct* allocate_mesh(double a, double b, double T, double BC_a, double BC_b, int t_res, int x_res);
void print_mesh(MeshStruct* mesh);
void initialize_mesh(MeshStruct* mesh);
double init_cond(MeshStruct* mesh, double x);
double index_to_xval(MeshStruct* mesh, int i);
void solve_ivp_dirichlet(MeshStruct* mesh);
void solve_index(MeshStruct* mesh, int t, int x);

// Main function
// -------------
int main(int argc, char *argv[]) {
    double a = 0;
    double b = 1;
    double T = 1;
    double BC_a = 0;
    double BC_b = 0;
    int t_res = 10;
    int x_res = 10;

    MeshStruct* mesh = allocate_mesh(a, b, T, BC_a, BC_b, t_res, x_res);

    // print_mesh(mesh);

    initialize_mesh(mesh);

    print_mesh(mesh);

    solve_ivp_dirichlet(mesh);

    print_mesh(mesh);

    return 1;
}


// Function definitions
// --------------------

// Allocates MeshStruct
MeshStruct* allocate_mesh(double a, double b, double T, double BC_a, double BC_b, int t_res, int x_res) {
    MeshStruct* mesh = malloc(sizeof(MeshStruct));

    mesh->a = a;
    mesh->b = b;
    mesh->T = T;
    mesh->BC_a = BC_a;
    mesh->BC_b = BC_b;
    mesh->t_res = t_res;
    mesh->x_res = x_res;

    mesh->mesh_arr = malloc(sizeof(double) * t_res);
    for (int t = 0; t < t_res; t++) {
        mesh->mesh_arr[t] = malloc(sizeof(double) * x_res);
    }

    return mesh;
}

// Prints the mesh size and the mesh array
void print_mesh(MeshStruct* mesh) {
    int t_res = mesh->t_res;
    int x_res = mesh->x_res;
    printf("Mesh Size: %d(t) x %d(x)\n", t_res, x_res);
    for(int t = t_res - 1; t >= 0; t--) {
        for (int x = 0; x < x_res; x++) {
            printf("%.2f ", mesh->mesh_arr[t][x]);
        }
        printf("\n");
    }
}

// Initializes the t=0 row with the initial condition
void initialize_mesh(MeshStruct* mesh) {
    double a = mesh->a;
    double b = mesh->b;
    int x_res = mesh->x_res;
    for(int i = 0; i < x_res; i++) {
        double x = index_to_xval(mesh, i);
        mesh->mesh_arr[0][i] = init_cond(mesh, x);
    }
}

// Stores the function for the initial condition
double init_cond(MeshStruct* mesh, double x) {
    double a = mesh->a;
    double b = mesh->b;
    return (a - x)*(x - b);
}

// Converts an x-index to the corresponding value between a and b
double index_to_xval(MeshStruct* mesh, int i) {
    double a = mesh->a;
    double b = mesh->b;
    int x_res = mesh->x_res;
    double Dx = (b - a) / (x_res - 1);
    return a + Dx*i;
}

void solve_ivp_dirichlet(MeshStruct* mesh) {
    double** mesh_arr = mesh->mesh_arr;
    int t_res = mesh->t_res;
    int x_res = mesh->x_res;
    double BC_a = mesh->BC_a;
    double BC_b = mesh->BC_b;

    for (int t = 1; t < t_res; t++) {
        // Set Dirichlet BCs
        mesh_arr[t][0] = BC_a;
        mesh_arr[t][x_res] = BC_b;

        // Fill in center from left to right
        for (int x = 1; x < x_res - 1; x++) {
            solve_index(mesh, t, x);
        }
    }

    return;
}

void solve_index(MeshStruct* mesh, int t, int x) {

}