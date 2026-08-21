#pragma once

#if __has_include(<mpi.h>) && !defined(LOB_FORCE_MPI_STUB)
#include <mpi.h>
#define LOB_HAS_REAL_MPI 1
#if (defined(MPI_VERSION) && MPI_VERSION >= 4) \
    || (defined(OMPI_MAJOR_VERSION) && OMPI_MAJOR_VERSION >= 5)
// Open MPI 5 provides the MPI-4 persistent collective entry points while its
// compatibility header still advertises MPI_VERSION == 3 and
// MPI_SUBVERSION == 1. Detect that implementation explicitly instead of
// silently disabling a function that is present in both the header and ABI.
#define LOB_HAS_MPI_PERSISTENT_COLLECTIVES 1
#else
#define LOB_HAS_MPI_PERSISTENT_COLLECTIVES 0
#endif
#else
#define LOB_HAS_REAL_MPI 0
#define LOB_HAS_MPI_PERSISTENT_COLLECTIVES 0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

using MPI_Comm = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Request = int;

struct MPI_Status {
    int MPI_SOURCE = 0;
    int MPI_TAG = 0;
    int _count = 0;
};

inline constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline constexpr MPI_Comm MPI_COMM_NULL = -1;
inline constexpr MPI_Datatype MPI_BYTE = 1;
inline constexpr MPI_Datatype MPI_INT = 2;
inline constexpr MPI_Datatype MPI_LONG_LONG = 3;
inline constexpr MPI_Datatype MPI_DOUBLE = 4;
inline constexpr MPI_Datatype MPI_UNSIGNED_LONG_LONG = 5;
inline constexpr MPI_Datatype MPI_UINT64_T = MPI_UNSIGNED_LONG_LONG;
inline constexpr MPI_Op MPI_SUM = 1;
inline constexpr MPI_Op MPI_MAX = 2;
inline constexpr MPI_Op MPI_MIN = 3;
inline constexpr int MPI_SUCCESS = 0;
inline constexpr int MPI_MAX_PROCESSOR_NAME = 256;
inline constexpr int MPI_ANY_SOURCE = -1;
inline constexpr int MPI_ANY_TAG = -1;
inline constexpr int MPI_UNDEFINED = -32766;
inline constexpr int MPI_THREAD_SINGLE = 0;
inline constexpr int MPI_THREAD_FUNNELED = 1;
inline constexpr int MPI_THREAD_SERIALIZED = 2;
inline constexpr int MPI_THREAD_MULTIPLE = 3;
inline constexpr MPI_Request MPI_REQUEST_NULL = 0;

inline MPI_Status mpi_status_ignore_storage{};
inline MPI_Status* MPI_STATUS_IGNORE = &mpi_status_ignore_storage;
inline MPI_Status* MPI_STATUSES_IGNORE = &mpi_status_ignore_storage;

inline std::size_t mpi_stub_type_size(MPI_Datatype type) {
    switch (type) {
        case MPI_INT: return sizeof(int);
        case MPI_LONG_LONG: return sizeof(long long);
        case MPI_DOUBLE: return sizeof(double);
        case MPI_UNSIGNED_LONG_LONG: return sizeof(unsigned long long);
        case MPI_BYTE:
        default: return 1;
    }
}

inline int MPI_Init(int*, char***) { return MPI_SUCCESS; }
inline int MPI_Init_thread(int*, char***, int required, int* provided) {
    if (provided) *provided = required;
    return MPI_SUCCESS;
}
inline int MPI_Query_thread(int* provided) {
    if (provided) *provided = MPI_THREAD_FUNNELED;
    return MPI_SUCCESS;
}
inline int MPI_Finalize() { return MPI_SUCCESS; }
inline int MPI_Abort(MPI_Comm, int) { return MPI_SUCCESS; }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = 0; return MPI_SUCCESS; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size = 1; return MPI_SUCCESS; }
inline int MPI_Comm_split(MPI_Comm, int color, int, MPI_Comm* new_comm) {
    *new_comm = color == MPI_UNDEFINED ? MPI_COMM_NULL : 0;
    return MPI_SUCCESS;
}
inline int MPI_Comm_free(MPI_Comm* comm) { *comm = MPI_COMM_NULL; return MPI_SUCCESS; }
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
inline int MPI_Send(const void*, int, MPI_Datatype, int, int, MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Recv(void*, int count, MPI_Datatype, int source, int tag, MPI_Comm, MPI_Status* status) {
    if (status && status != MPI_STATUS_IGNORE) {
        status->MPI_SOURCE = source < 0 ? 0 : source;
        status->MPI_TAG = tag;
        status->_count = count;
    }
    return MPI_SUCCESS;
}
inline int MPI_Isend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request* request) {
    if (request) *request = 1;
    return MPI_SUCCESS;
}
inline int MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request* request) {
    if (request) *request = 1;
    return MPI_SUCCESS;
}
inline int MPI_Test(MPI_Request* request, int* complete, MPI_Status*) {
    if (complete) *complete = 1;
    if (request) *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
}
inline int MPI_Wait(MPI_Request* request, MPI_Status*) {
    if (request) *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
}
inline int MPI_Waitall(int count, MPI_Request* requests, MPI_Status*) {
    for (int i = 0; i < count; ++i) requests[i] = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
}
inline int MPI_Probe(int source, int tag, MPI_Comm, MPI_Status* status) {
    if (status) {
        status->MPI_SOURCE = source < 0 ? 0 : source;
        status->MPI_TAG = tag;
        status->_count = 0;
    }
    return MPI_SUCCESS;
}
inline int MPI_Iprobe(int source, int tag, MPI_Comm, int* available, MPI_Status* status) {
    if (available) *available = 0;
    if (status) {
        status->MPI_SOURCE = source < 0 ? 0 : source;
        status->MPI_TAG = tag;
        status->_count = 0;
    }
    return MPI_SUCCESS;
}
inline int MPI_Get_count(const MPI_Status* status, MPI_Datatype, int* count) {
    if (count) *count = status ? status->_count : 0;
    return MPI_SUCCESS;
}
inline int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf,
                    static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, const int*, const int* displs, MPI_Datatype recvtype,
                       int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf && sendcount > 0) {
        const int displacement = displs ? displs[0] : 0;
        std::memcpy(static_cast<unsigned char*>(recvbuf)
                        + static_cast<std::size_t>(displacement) * mpi_stub_type_size(recvtype),
                    sendbuf,
                    static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf,
                    static_cast<std::size_t>(sendcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Scatterv(const void* sendbuf, const int*, const int* displs, MPI_Datatype sendtype,
                        void* recvbuf, int recvcount, MPI_Datatype, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf && recvcount > 0) {
        const int displacement = displs ? displs[0] : 0;
        std::memcpy(recvbuf,
                    static_cast<const unsigned char*>(sendbuf)
                        + static_cast<std::size_t>(displacement) * mpi_stub_type_size(sendtype),
                    static_cast<std::size_t>(recvcount) * mpi_stub_type_size(sendtype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                      MPI_Op, int root, MPI_Comm) {
    if (root == 0 && recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf,
                    static_cast<std::size_t>(count) * mpi_stub_type_size(datatype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                         MPI_Op, MPI_Comm) {
    if (recvbuf && sendbuf) {
        std::memcpy(recvbuf, sendbuf,
                    static_cast<std::size_t>(count) * mpi_stub_type_size(datatype));
    }
    return MPI_SUCCESS;
}
inline int MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                          MPI_Request* request) {
    const int status = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    if (request) *request = status == MPI_SUCCESS ? 1 : MPI_REQUEST_NULL;
    return status;
}
inline int MPI_Request_free(MPI_Request* request) {
    if (request) *request = MPI_REQUEST_NULL;
    return MPI_SUCCESS;
}
#endif
