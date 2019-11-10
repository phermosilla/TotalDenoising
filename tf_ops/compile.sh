rm build/*
/usr/local/cuda/bin/nvcc -std=c++11  knn.cu -I../MCCNN/tf_ops -o build/knn.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  neighbor_rand_select.cu -I../MCCNN/tf_ops -o build/neighbor_rand_select.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  point_to_mesh_distance.cu -I../MCCNN/tf_ops -o build/point_to_mesh_distance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -std=c++11  spatial_conv_gauss.cu -I../MCCNN/tf_ops -o build/spatial_conv_gauss.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 build/knn.cu.o  build/neighbor_rand_select.cu.o  build/point_to_mesh_distance.cu.o  build/spatial_conv_gauss.cu.o knn.cc neighbor_rand_select.cc point_to_mesh_distance.cc spatial_conv_gauss.cc -I../MCCNN/tf_ops -o build/tf_ops_module.so -shared -fPIC -I/home/peter/.local/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -L/home/peter/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2
