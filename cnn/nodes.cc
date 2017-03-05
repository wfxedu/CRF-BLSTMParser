#include "cnn/nodes.h"

#include <limits>
#include <cmath>

#include "cnn/functors.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#endif

using namespace std;

// notes on implementing differentiable components
// 1) fx can be understood as a pointer to the (preallocated) location for the result
//    of forward to be stored
// 2) fx is not initialized, so after calling forward fx must point to the correct answer
// 3) fx can be repointed to an input, if forward(x) evaluates to x (e.g., in reshaping)
// 4) dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on
//    the same x_i. Even, e.g., Identity must be implemented as
//    dEdx1 += dEdf. THIS IS EXTREMELY IMPORTANT
// 5) scalars results of forward are placed in fx.v[0]
// 6) CNN manages its own memory, not Eigen, and it is configured with the
//    EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate
//    memory, it is (probably) because of an implicit creation of a temporary variable.
//    To tell Eigen this is not necessary, the noalias() method is available. If you really
//    do need a temporary variable, its capacity must be requested by Node::aux_storage_space
//
// notes on debugging problems with differentiable components
// 1) fx is uninitialized when forward is called- are you relying on it being 0?
// 2) dEdxi must accummulate (see point 4 above!)
//

namespace cnn {

size_t Min::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Min::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto y = *fx;
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  Tensor t(fx.d, static_cast<float*>(aux_mem));
  auto u = *t;
  u = (x1.array() < x2.array()).matrix().cast<float>();
  y = x1.cwiseMin(x2);
}

void Min::backward(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem));
  if (i == 0) {
    *dEdxi += (*t).cwiseProduct(*dEdf);
  } else {
    *dEdxi += (*t).binaryExpr(*dEdf, FMaxBackwardInv());
  }
}

size_t Max::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Max::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto y = *fx;
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  Tensor t(fx.d, static_cast<float*>(aux_mem));
  auto u = *t;
  u = (x1.array() > x2.array()).matrix().cast<float>();
  y = x1.cwiseMax(x2);
}

void Max::backward(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem));
  if (i == 0) {
    *dEdxi += (*t).cwiseProduct(*dEdf);
  } else {
    *dEdxi += (*t).binaryExpr(*dEdf, FMaxBackwardInv());
  }
}

void TraceOfProduct::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 * x2.transpose()).trace();
}

void TraceOfProduct::backward(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  assert(i < 2);
  const float d = dEdf.v[0];
  auto xother = **xs[1 - i];
  *dEdxi += d * xother;
}

void ConstScalarMultiply::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  *fx = (**xs[0]) * alpha;
}

void ConstScalarMultiply::backward(const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  assert(i == 0);
  *dEdxi += *dEdf * alpha;
}

void DotProduct::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  *fx = (**xs[0]).transpose() * (**xs[1]);
}

void DotProduct::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  (*dEdxi) += (dEdf.v[0]) * (**xs[1 - i]);
}

void Transpose::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.rows() == 1 || dim.cols() == 1) {
    fx.v = xs[0]->v;
  } else {
#if HAVE_CUDA
    CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, fx.d.rows(), fx.d.cols(),
                             kSCALAR_ONE, xs[0]->v, xs[0]->d.rows(), kSCALAR_ZERO, NULL, fx.d.rows(), fx.v, fx.d.rows()));
#else
    *fx = (**xs[0]).transpose();
#endif
  }
}

void Transpose::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, dEdxi.d.rows(), dEdxi.d.cols(),
                           kSCALAR_ONE, dEdf.v, dEdf.d.rows(), kSCALAR_ONE, dEdxi.v, dEdxi.d.rows(), dEdxi.v, dEdxi.d.rows()));
#else
  *dEdxi += (*dEdf).transpose();
#endif
}

void Reshape::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  // just point to the input memory and change dimensions
  // dimensions are handled by forward_dim
  fx.v = xs[0]->v;
}

void Reshape::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  const Tensor reshaped(dEdxi.d, dEdf.v);
  *dEdxi += *reshaped;
}

void SumColumns::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  auto y = *fx;
  if (xs.size() == 1) {
    y = x.rowwise().sum();
  } else {
    throw std::invalid_argument("two inputs in SumColumns::forward!");
  }
}

void SumColumns::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  auto out = *dEdxi;
  // this uses Eigen's broadcast capability
  // the following doesn't compile, so i use the next line
  //out.colwise() += *dEdf;
  out.colwise() += (*dEdf).col(0);
}

void KMHNGram::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  const int new_cols = x.cols() - n + 1;
  assert(new_cols > 0);
  auto res = *fx;
  res.setZero();
  for (int j = 0; j < new_cols; ++j) {
    auto c_j = res.col(j);
    for (unsigned k = 0; k < n; ++k)
      c_j += x.col(j + k);
  }
}

void KMHNGram::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (*dEdxi).col(j+k) += (*dEdf).col(j);
}

//   Y_ij = A_ijk * B_k (+ C_ij)
void InnerProduct3D_1D::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto b = **xs[1];
  auto y = *fx;
  const int i = y.rows();
  const int j = y.cols();
  const int k = b.rows();
  // the following reshape tensors into order 1 or 2 sizes
  // but they point to the same memory
  Tensor ta({i*j,k}, xs[0]->v);
  Tensor ty({i*j}, fx.v);
  auto A = *ta;
  if (xs.size() == 3) {
    Tensor tc({i*j}, xs[2]->v);
    auto c = *tc;
    // want to do A * b + c, but it triggers memory allocation
    (*ty) = c;
    (*ty).noalias() += A * b;
  } else {
    assert(xs.size() == 2);
    (*ty).noalias() = A * b;
  }
}

void InnerProduct3D_1D::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto b = **xs[1];
  auto y = *fx;
  const int si = y.rows();
  const int sj = y.cols();
  const int sk = b.rows();
  Tensor tdEdf({si*sj}, dEdf.v);
  if (i == 0) { // 3-tensor
    Tensor tdEdxi({si*sj, sk}, dEdxi.v);
    (*tdEdxi).noalias() += *tdEdf * (**xs[1]).transpose();
  } else if (i == 1) { // vector
    Tensor ta({si*sj,sk}, xs[0]->v);
    (*dEdxi).noalias() += (*ta).transpose() * *tdEdf;
  } else { // matrix bias
    *dEdxi += *dEdf;
  }
}

size_t GaussianNoise::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void GaussianNoise::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor m(dim, (float*)aux_mem);
  TensorTools::RandomizeNormal(0, stddev, m);
  (*fx) = **xs[0] + *m;
}

void GaussianNoise::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

size_t Dropout::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Dropout::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
	if (cnn_train_mode) {
		Tensor m(dim, (float*)aux_mem);
		TensorTools::RandomBernoulli(m, (1.f - p), 1.0f);//1.f / (1.f - p)
		(*fx) = (**xs[0]).cwiseProduct(*m);
	}
	else {
		float*v = (float*)aux_mem;
		v[0] = (1.f - p);
		(*fx) = (**xs[0])*v[0];
	}
}

void Dropout::backward(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {

	if (cnn_train_mode) {
		Tensor m(dim, (float*)aux_mem);
		(*dEdxi) += (*dEdf).cwiseProduct(*m);
	}
	else {
		float*v = (float*)aux_mem;
		(*dEdxi) += (*dEdf)*v[0];
	}
}

size_t BlockDropout::aux_storage_size() const {
  // we just need to remember whether this entire block is turned on (1.0) or off (0.0)
  return 1 * sizeof(float);
}

void BlockDropout::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  bernoulli_distribution distribution(1.0 - dropout_probability);
  float block_multiplier = distribution(*rndeng)? 1.0 : 0.0;
  block_multiplier = 
    dropout_probability == 1.0? 0.0 : block_multiplier / (1.0 - dropout_probability);
  if (dropout_probability > 1.0 || dropout_probability < 0.0) {
    assert(false && "dropout probability must be in the range [0, 1]");
  }
  *(static_cast<float*>(aux_mem)) = block_multiplier;
  (*fx) = **xs[0] * block_multiplier;
}

void BlockDropout::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  float block_multiplier = *(static_cast<float*>(aux_mem));
  (*dEdxi) += (*dEdf) * block_multiplier;
}

void ConstantPlusX::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.unaryExpr(FConstantPlus(c));
}

void ConstantPlusX::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

void ConstantMinusX::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vconstant_minusx(fx.d.size(), c, xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FConstantMinus(c));
#endif
}

void ConstantMinusX::backward(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vnegate_backward(dEdxi.d.size(), dEdf.v, dEdxi.v);
#else
  *dEdxi -= *dEdf;
#endif
}

template <class T>
EIGEN_STRONG_INLINE float logsumexp(const T& x) {
  const float m = x.maxCoeff();
  float z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

// this i need to do something better, but this is a work-around
// if this is too small, just make it bigger
#define MAX_LOG_SUM_EXP 65536
size_t LogSumExp::aux_storage_size() const {
  return MAX_LOG_SUM_EXP * sizeof(float);
}

void LogSumExp::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
  for (unsigned i = 0; i < xs.size(); ++i)
    static_cast<float*>(aux_mem)[i] = (**xs[i])(0,0);
  Dim r = {(int)xs.size()};
  Tensor v(r, static_cast<float*>(aux_mem));
  fx.v[0] = logsumexp(*v);
}

void LogSumExp::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  if (xs.size() == 0) {
    *dEdxi += *dEdf;
    return;
  }
  // df/dx_i = 1/{sum_j exp(x_j)} * exp(x_i)}
  //         = 1/{exp f(x)} * exp(x_i)
  //         = exp(x_i - f(x))
  auto d = *dEdxi;
  d.array() += (**xs[i] - *fx).array().exp() * (*dEdf).array();
}

void Sum::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
#if HAVE_CUDA
  TensorTools::Zero(fx);
  for (unsigned i = 0; i < num_args; ++i)
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[i]->v, 1, fx.v, 1));
#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = **xs[0]; break;
    case 2: res = **xs[0] + **xs[1]; break;
    case 3: res = **xs[0] + **xs[1] + **xs[2]; break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += **xs[i] + **xs[i+1] + **xs[i+2] + **xs[i+3];
#endif
}

void Sum::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {

#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, dEdf.v, 1, dEdxi.v, 1));
#else
  *dEdxi += *dEdf;
#endif
}

void Average::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = **xs[0]; break;
    case 2: res = **xs[0] + **xs[1]; break;
    case 3: res = **xs[0] + **xs[1] + **xs[2]; break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += **xs[i] + **xs[i+1] + **xs[i+2] + **xs[i+3];
  res /= num_args;
}

void Average::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += (*dEdf / xs.size());
}

void Tanh::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vtanh(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  (*fx).array() = x.array().tanh();
#endif
}

void Tanh::backward(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vtanh_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FTanhBackward());
#endif
}

void Square::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  (*fx).array() = x.array().square();
}

void Square::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseProduct(x) * 2;
}

void Cube::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  (*fx).array() = x.array().cube();
}

void Cube::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  auto x = **xs[0];
//  *dEdxi += (*dEdf).cwiseProduct(x.cwiseProduct(x)) * 3;
  (*dEdxi).array() += (*dEdf).array() * x.array().square() * 3;
}

void Exp::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.array().exp();
}

void Exp::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += (*dEdf).cwiseProduct(*fx);
}

void Log::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.array().log();
}

void Log::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseQuotient(x);
}

void Concatenate::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned rows = 0;
  for (auto x : xs) rows += x->d.rows();
  // the following should use auxiliary memory
  src_row_indices.resize(xs.size());
  unsigned ind = 0;
  unsigned k = 0;
  for (auto x : xs) {
    src_row_indices[k++] = ind;
    auto & xi = *x;
    const unsigned rows = xi.d.rows();
#if HAVE_CUDA
    assert(xi.d.cols() == 1); // this can be relaxed to the same everywhere
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[ind], &xi.v[0], sizeof(float) * rows, cudaMemcpyDeviceToDevice));
#else
    (*fx).middleRows(ind, rows) = *xi;
#endif
    ind += rows;
  }
}

void Concatenate::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < src_row_indices.size());
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = src_row_indices[i];
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows, kSCALAR_ONE, &dEdf.v[begin], 1, dEdxi.v, 1));
#else
  *dEdxi += (*dEdf).middleRows(begin, rows);
#endif
}

#define MAX_CONCAT_COLS_ARGS 512
size_t ConcatenateColumns::aux_storage_size() const {
  return MAX_CONCAT_COLS_ARGS * sizeof(unsigned);
}

void ConcatenateColumns::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned c = 0;
  assert(xs.size() < MAX_CONCAT_COLS_ARGS);
  for (unsigned i = 0; i < xs.size(); ++i) {
    static_cast<unsigned*>(aux_mem)[i] = c;
#if HAVE_CUDA
    assert(xs[i]->d.cols() == 1);
    // CUBLAS matricies are column-major, so just copy the memory
    auto & xi = *xs[i];
    const unsigned rows = xi.d.rows();
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[i*rows], &xi.v[0], sizeof(float) * rows, cudaMemcpyDeviceToDevice));
#else
    auto xi = **xs[i];
    int d = xi.cols();
    (*fx).middleCols(c, d) = xi;
    c += d;
#endif
  }
}

void ConcatenateColumns::backward(const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
#if HAVE_CUDA
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = i*rows;
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows, kSCALAR_ONE, &dEdf.v[begin], 1, dEdxi.v, 1));
#else
  auto dEdx = *dEdxi;
  int d = dEdx.cols();
  int c = static_cast<unsigned*>(aux_mem)[i];
  dEdx += (*dEdf).middleCols(c, d);
#endif
}

void PairwiseRankLoss::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss(fx.d.size(), margin, xs[0]->v, xs[1]->v, fx.v);
#else
  auto a = **xs[0];
  auto b = **xs[1];
  *fx = a.binaryExpr(b, FPairwiseRankLoss(margin));
#endif
}

void PairwiseRankLoss::backward(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss_backward(dEdf.d.size(), (i == 0), fx.v, dEdf.v, dEdxi.v);
#else
  if (i == 0) {
    *dEdxi -= (*fx).binaryExpr(*dEdf, FRectifyBackward());
  } else {
    *dEdxi += (*fx).binaryExpr(*dEdf, FRectifyBackward());
  }
#endif
}

size_t Hinge::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Hinge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  const unsigned rows = x.rows();
  float y = 0;
  float* eloss = static_cast<float*>(aux_mem);
  const real mlystar = margin - x(*pelement);
  for (unsigned i = 0; i < rows; ++i) {
    if (*pelement != i) {
      eloss[i] = max(0.f, mlystar + x(i));
      y += eloss[i];
    } else {
      eloss[i] = 0;
    }
  }
  fx.v[0] = y;
}

void Hinge::backward(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
  assert(i == 0);
  if (fx.v[0]) { // there was some loss
    const float d = dEdf.v[0];
    const unsigned rows = dEdxi.d.rows();
    const float* eloss = static_cast<const float*>(aux_mem);
    unsigned tne = 0;  // total number of errors
    for (unsigned i = 0; i < rows; ++i)
      if (eloss[i] > 0) {
        (*dEdxi)(i) += d;
        ++tne;
      }
    (*dEdxi)(*pelement) -= d * tne;
  }
}

void Identity::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.d = xs[0]->d;
  fx.v = xs[0]->v;
}

void Identity::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

void MaxPooling1D::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL5\n"; abort();
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  assert(x.cols() == 1);
  const unsigned fx_rows = x_rows / width;
  ind.resize(fx_rows);
  Tensor fx = Zero(Dim(fx_rows, 1));
  for (unsigned i = 0; i < fx_rows; ++i) {
    unsigned from = i * width;
    unsigned to = from + width;
    if (to > x_rows) to = x_rows;
    real best = x(from, 0);
    unsigned bestr = from;
    for (unsigned r = from + 1; r < to; ++r) {
      if (x(r, 0) > best) {
        best = x(r,0);
        bestr = r;
      }
    }
    ind[i] = bestr;
    fx(i, 0) = best;
  }
  return fx;
#endif
}

void MaxPooling1D::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  cerr << "FIX IMPL6\n"; abort();
#if 0
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  Tensor dEdx = Zero(Dim(x_rows, 1));
  const unsigned fx_rows = x_rows / width;
  assert(fx_rows == ind.size());
  assert(fx_rows == dEdf.rows());
  for (unsigned i = 0; i < fx_rows; ++i)
    dEdx(ind[i], 0) = dEdf(i, 0);
  return dEdx;
#endif
}

void Softmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    gpu::softmax(xs[0]->d.size(), xs[0]->v, fx.v);
#else
    auto x = **xs[0];
    *fx = x.unaryExpr(FSoftmaxNormalize(logsumexp(x)));
#endif
  } else {
    cerr << "SoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

void Softmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::softmax_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  float off_diag_sum = -(*fx).cwiseProduct(*dEdf).sum();
  *dEdxi += (*fx).binaryExpr(*dEdf, FSoftmaxBackward(off_diag_sum));
#endif
}

void PickNegLogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    logz = (float*)fxs->allocate(sizeof(float));
#if HAVE_CUDA
    gpu::pnlsoftmax(xs[0]->d.size(), *pval, xs[0]->v, fx.v, logz);
#else
    auto x = **xs[0];
    *logz = logsumexp(x);
    fx.v[0] = *logz - x(*pval);
#endif
  } else {
    cerr << "SoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

void PickNegLogSoftmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
    const auto elem = *pval;
#if HAVE_CUDA
    gpu::pnlsoftmax_backward(dEdxi.d.size(), elem, xs[0]->v, dEdf.v, logz, dEdxi.v);
#else
    const float err = dEdf.v[0];
    auto x = **xs[0];
    // logz is computed in the forward pass and cached
    *dEdxi += x.unaryExpr(FNegLogSoftmaxBackward(*logz, err));
    (*dEdxi)(elem) -= err;
#endif
  } else {
    cerr << "PickNegLogSoftmax not implemented for multiple columns\n";
    abort();
  }
}

void LogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  if (xs[0]->d.cols() == 1) {
    auto x = **xs[0];
    *fx = x.unaryExpr(FLogSoftmaxNormalize(logsumexp(x)));
  } else {
    cerr << "LogSoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

void LogSoftmax::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
    float off_diag_sum = -(*fx).binaryExpr(*dEdf, FWeightedError()).sum();
    *dEdxi += (*fx).binaryExpr(*dEdf, FLogSoftmaxBackward(off_diag_sum));
  } else {
    cerr << "LogSoftmaxBackward not implemented for multiple columns\n";
    abort();
  }
}

template <class T>
EIGEN_STRONG_INLINE real logsumexp(const T& x, const vector<unsigned>& denom) {
  real m = x(denom[0],0);
  for (auto i : denom) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (auto i : denom)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

void RestrictedLogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  // TODO create auxiliary mask with -infty's
  // and do usual LogSoftmax stuff
  assert(xs.size() == 1);
  assert(denom.size() > 0);
  auto x = **xs[0];
  assert(x.cols() == 1);
  const real logz = logsumexp(x, denom);
  TensorTools::Constant(fx, -numeric_limits<real>::infinity());
  for (auto i : denom)
    (*fx)(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (*fx)(denom.front(), 0) = 0;
}

void RestrictedLogSoftmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  assert(i == 0);
  float z = 0;
  for (auto ind : denom)
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
}

// x_1 is a vector
// y = (x_1)_{*pval}
void PickElement::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  fx.v[0] = x(*pval);
}

// derivative is 0 in all dimensions except 1 for the selected element
void PickElement::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
  (*dEdxi)(*pval) += dEdf.v[0];
}

// x_1 is a vector
// y = (x_1)[start:end]
// slice of vector from index start (inclusive) to index end (exclusive)
void PickRange::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  assert(x.cols() == 1);
  assert(start >= 0);
  assert(end <= x.rows());
  assert(start < end);
  assert(int(fx.d.rows()) == int(end-start));
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpyAsync(&fx.v[0], &xs[0]->v[start], sizeof(float) * (end-start), cudaMemcpyDeviceToDevice));
#else
  (*fx) = x.block(start, 0, end-start, 1);
#endif
}

// derivative is 0 in all dimensions except the slice range
void PickRange::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
  assert(int(dEdf.d.rows()) == int(end-start));
  assert(dEdf.d.cols() == 1);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, end-start, kSCALAR_ONE, dEdf.v, 1, &dEdxi.v[start], 1));
#else
  (*dEdxi).block(start, 0, end-start, 1) += (*dEdf);
#endif
}

#if HAVE_CUDA
inline void CUDAMatrixMultiply(const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {
  if (r.d.ndims() == 1 || r.d.cols() == 1) {
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, l.d.rows(), l.d.cols(),
               kSCALAR_ONE, l.v, l.d.rows(), r.v, 1, acc_scalar, y.v, 1));
  } else {
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols(), l.d.cols(),
          kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          acc_scalar, y.v, y.d.rows()));
  }
}
#endif

void MatrixMultiply::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  // fx = 0*fx + xs[0] * xs[1]
  CUDAMatrixMultiply(*xs[0], *xs[1], fx, kSCALAR_ZERO);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  (*fx).noalias() = x1 * x2;
#endif
}

void MatrixMultiply::backward(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  if (i == 0) {
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
          dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
          kSCALAR_ONE,
          dEdf.v, dEdf.d.rows(),
          xs[1]->v, xs[1]->d.rows(),
          kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
  } else {
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dEdxi.d.rows(), dEdxi.d.cols(), xs[0]->d.rows(),
          kSCALAR_ONE,
          xs[0]->v, xs[0]->d.rows(),
          dEdf.v, xs[0]->d.rows(),
          kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
  }
#else
  if (i == 0) {
    (*dEdxi).noalias() += *dEdf * (**xs[1]).transpose();
  } else {
    (*dEdxi).noalias() += (**xs[0]).transpose() * *dEdf;
  }
#endif
}

void CwiseQuotient::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  *fx = x1.cwiseQuotient(x2);
}

void CwiseQuotient::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
    auto x2 = **xs[1];
    *dEdxi += (*dEdf).cwiseQuotient(x2);
  } else { // i = 1
    auto x1 = **xs[0];
    auto x2 = **xs[1];
    *dEdxi -= (*dEdf).cwiseQuotient(x2.cwiseProduct(x2)).cwiseProduct(x1);
  }
}

void CwiseMultiply::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::vcwise_product(fx.d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  *fx = x1.cwiseProduct(x2);
#endif
}

void CwiseMultiply::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[1]->v, dEdxi.v);
#else
    auto x2 = **xs[1];
    *dEdxi += (*dEdf).cwiseProduct(x2);
#endif
  } else {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[0]->v, dEdxi.v);
#else
    auto x1 = **xs[0];
    *dEdxi += (*dEdf).cwiseProduct(x1);
#endif
  }
}

void AffineTransform::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() % 2 == 1);
  if (xs.size() == 1) {
    fx.v = xs[0]->v;
    return;
  } else {
#if HAVE_CUDA
    for (unsigned i = 1; i < xs.size(); i += 2)
      // fx = (acc_sclar)*fx + xs[0] * xs[1]
      CUDAMatrixMultiply(*xs[i], *xs[i + 1], fx, (i == 1) ? kSCALAR_ZERO : kSCALAR_ONE);
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[0]->v, 1, fx.v, 1));
#else
    if (xs.size() == 3) {
      // size 3 is optimized in newer versions of eigen
      (*fx).noalias() = **xs[0] + **xs[1] * **xs[2];
    } else {
      (*fx) = **xs[0];
      for (unsigned i = 1; i < xs.size(); i += 2)
        (*fx).noalias() += (**xs[i]) * (**xs[i + 1]);
    }
#endif
  }
}

void AffineTransform::backward(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(i < xs.size());
  if (i == 0) { // bias term
#if HAVE_CUDA
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, dEdxi.d.size(), kSCALAR_ONE, dEdf.v, 1, dEdxi.v, 1));
#else
    *dEdxi += *dEdf;
#endif
  } else if (i % 2 == 1) { // left argument of matrix multiply
#if HAVE_CUDA
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
          dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
          kSCALAR_ONE,
          dEdf.v, dEdf.d.rows(),
          xs[i+1]->v, xs[i+1]->d.rows(),
          kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
    (*dEdxi).noalias() += *dEdf * (**xs[i+1]).transpose();
#endif
  } else {  // right argument of matrix multiply
#if HAVE_CUDA
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dEdxi.d.rows(), dEdxi.d.cols(), xs[i-1]->d.rows(),
          kSCALAR_ONE,
          xs[i-1]->v, xs[i-1]->d.rows(),
          dEdf.v, xs[i-1]->d.rows(),
          kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
    (*dEdxi).noalias() += (**xs[i-1]).transpose() * *dEdf;
#endif
  }
}

void Negate::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vnegate(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = -x;
#endif
}

void Negate::backward(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  gpu::vnegate_backward(fx.d.size(), dEdf.v, dEdxi.v);
#else
  *dEdxi -= *dEdf;
#endif
}

void Rectify::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vrelu(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FRectify());
#endif
}

void Rectify::backward(const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vrelu_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FRectifyBackward());
#endif
}

void HuberDistance::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
  auto x = *xs[0];
  auto y = *xs[1];
  const FHuberForward fhf(d);
  const size_t s = x.d.size();
  float dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += fhf(x.v[i] - y.v[i]);
  fx.v[0] = dist;
}

void HuberDistance::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
  auto x = **xs[i];
  auto y = **xs[1-i];
  *dEdxi += (x - y).unaryExpr(FHuberBackward(d, dEdf.v[0]));
}

void L1Distance::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
  auto x = **xs[0];
  auto y = **xs[1];
  fx.v[0] = (x - y).lpNorm<1>();
}

void L1Distance::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
  auto x = **xs[i];
  auto y = **xs[1-i];
  *dEdxi += (x - y).unaryExpr(FL1Backward(dEdf.v[0]));
}

void PoissonRegressionLoss::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  const auto y = *pty;
  const auto z = lgamma(y + 1);
  const auto x = xs[0]->v[0];
  fx.v[0] = expf(x) + z - y * x;
}

void PoissonRegressionLoss::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  const auto x = xs[0]->v[0];
  const auto y = *pty;
  auto& dEdx = dEdxi.v[0];
  dEdx += expf(x) - y;
}

void SquaredEuclideanDistance::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::sqeucdist(xs[0]->d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 - x2).squaredNorm();
#endif
}

void SquaredEuclideanDistance::backward(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  gpu::sqeucdist_backward(xs[0]->d.size(), dEdf.v, xs[0]->v, xs[1]->v, dEdxi.v, i);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  real scale = dEdf.v[0] * 2;
  if (i == 1) scale = -scale;
  *dEdxi += scale * (x1 - x2);
#endif
}

void LogisticSigmoid::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vlogistic(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FLogisticSigmoid());
#endif
}

void LogisticSigmoid::backward(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vlogistic_backward(dEdf.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FLogisticSigmoidBackward());
#endif
}

void SoftSign::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  *fx = x.unaryExpr(FSoftSign());
}

void SoftSign::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  *dEdxi += (*fx).binaryExpr(*dEdf, FSoftSignBackward());
}

void BinaryLogLoss::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = *xs[0];
  auto y = *xs[1];
  FBinaryLogLoss bll;
  const size_t s = x.d.size();
  float dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += bll(x.v[i], y.v[i]);
  fx.v[0] = dist;
}

void BinaryLogLoss::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  *dEdxi += (**xs[i]).binaryExpr(**xs[1-i], FBinaryLogLossBackward(dEdf.v[0]));
}

} // namespace cnn
