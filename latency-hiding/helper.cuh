#define cudaErrCheck(call) {                                                  \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}                                                                             

