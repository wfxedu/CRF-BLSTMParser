#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {
extern bool cnn_train_mode;
void Initialize(int& argc, char**& argv, unsigned random_seed = 0);

} // namespace cnn

#endif
