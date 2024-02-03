#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void matrixThread(int size, __int64** Am, __int64** Bm, __int64** Rt)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                Rt[i][j] += Am[i][k] * Bm[k][j];
            }
        }
    }
}

int main() {
    int size = 4096;

    __int64** Am = new __int64* [size];
    __int64** Bm = new __int64* [size];
    __int64** Rs = new __int64* [size];
    __int64** Rv = new __int64* [size];
    __int64** Rt = new __int64* [size];
    __int64** Rtv = new __int64* [size];

    for (int i = 0; i < size; i++) {
        Am[i] = new __int64[size];
        Bm[i] = new __int64[size];
        Rs[i] = new __int64[size];
        Rv[i] = new __int64[size];
        Rt[i] = new __int64[size];
        Rtv[i] = new __int64[size];
        for (int j = 0; j < size; j++) {
            Am[i][j] = rand() % 10;
            Bm[i][j] = rand() % 10;
            Rs[i][j] = 0;
            Rv[i][j] = 0;
            Rt[i][j] = 0;
            Rtv[i][j] = 0;
        }
    }


    cout << "Scalar: \n";
    auto start = chrono::system_clock::now();

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                Rs[i][j] += Am[i][k] * Bm[k][j];
            }
        }
    }

    auto end = chrono::system_clock::now();
    chrono::duration<double> diff = end - start;
    cout << "Time to calculate " << diff.count() << " s" << endl;

    cout << "OpenMP: \n";

    start = chrono::system_clock::now();
    matrixThread(size, Am, Bm, Rt);
    end = chrono::system_clock::now();
    diff = end - start;
    cout << "Time to calculate " << diff.count() << " s" << endl;


    bool equal = true;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (Rs[i][j] != Rt[i][j]) {
                equal = false;
            }
        }
    }
    if (equal) {
        cout << "Matrix is equal\n";
    }
    else {
        cout << "Matrix not equal\n";
    }

    for (int i = 0; i < size; i++) {
        delete[] Am[i];
        delete[] Bm[i];
        delete[] Rs[i];
        delete[] Rv[i];
        delete[] Rt[i];
        delete[] Rtv[i];
    }

    delete[] Am;
    delete[] Bm;
    delete[] Rs;
    delete[] Rv;
    delete[] Rt;
    delete[] Rtv;

    return 0;
}
