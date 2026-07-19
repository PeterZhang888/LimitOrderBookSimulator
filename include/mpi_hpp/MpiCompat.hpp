#pragma once

#if __has_include(<mpi.h>) && !defined(LOB_FORCE_MPI_STUB)
#include <mpi.h>
#define LOB_HAS_REAL_MPI 1
#else
#define LOB_HAS_REAL_MPI 0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>

using MPI_Comm = int;
using MPI_Datatype = int;
using MPI_Op = int;

inline constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline constexpr MPI_Datatype MPI_BYTE = 1;
inline constexpr MPI_Datatype MPI_INT = 2;
inline constexpr MPI_Datatype MPI_LONG_LONG = 3;
inline constexpr MPI_Datatype MPI_DOUBLE = 4;
inline constexpr MPI_Op MPI_SUM = 1;
inline constexpr MPI_Op MPI_MAX = 2;
inline constexpr int MPI_SUCCESS = 0;
inline constexpr int MPI_MAX_PROCESSOR_NAME = 256;

inline std::size_t mpi_stub_type_size(MPI_Datatype type) {
    switch (type) {
        case MPI_INT: return sizeof(int);
        case MPI_LONG_LONG: return sizeof(long long);
        case MPI_DOUBLE: return sizeof(double);
        case MPI_BYTE:
        default: return 1;
    }
}

inline int MPI_Init(int*, char***) { return MPI_SUCCESS; }
inline int MPI_Finalize() { return MPI_SUCCESS; }
inline int MPI_Abort(MPI_Comm, int) { return MPI_SUCCESS; }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = 0; return MPI_SUCCESS; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size = 1; return MPI_SUCCESS; }
inline int MPI_Get_processor_name(char* name, int* length) {
    const char* value = "single-process";
    const std::size_t n = std::strlen(value);
    if (name) std::memcpy(name, value, n + 1);
    if (length) *length = static_cast<int>(n);
    return MPI_SUCCESS;
}
inline int MPI_Barrier(MPI_Comm) { return MPI_SUCCESS; }
inline double MPI_Wtime() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}
inline int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf, static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, const int*, const int* displs, MPI_Datatype,
                       int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf && sendcount > 0) {
        const int offset = displs ? displs[0] : 0;
        std::memcpy(static_cast<char*>(recvbuf) + offset,
                    sendbuf,
                    static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf, static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Scatterv(const void* sendbuf, const int*, const int* displs, MPI_Datatype sendtype,
                        void* recvbuf, int recvcount, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf && recvcount > 0) {
        const int offset = displs ? displs[0] : 0;
        std::memcpy(recvbuf,
                    static_cast<const char*>(sendbuf) + offset,
                    static_cast<std::size_t>(recvcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                      MPI_Op, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf, static_cast<std::size_t>(count) * mpi_stub_type_size(datatype));
    }
    return MPI_SUCCESS;
}
#endif
