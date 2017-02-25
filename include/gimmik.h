#ifdef __cplusplus
#define restrict
#endif

extern "C"
{
  void
  gimmik_mm_cpu(int m, int n, int k, const double alpha, 
           const double* restrict a, int lda,
           const double* restrict b, int ldb,
           const double beta, double* restrict c, int ldc,
           unsigned long id);
}

#ifdef _GPU
  void
  gimmik_mm_gpu(int m, int n, int k, const double alpha, 
           const double* restrict a, int lda,
           const double* restrict b, int ldb,
           const double beta, double* restrict c, int ldc,
           unsigned long id);
#endif
