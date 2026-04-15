# cuda_kernel_test

## compile

```bash
nvcc test.cu -o test
```

## nsys easy

```bash
nsys_easy -t cuda,osrt -s none -c none -o nsys_easy -r cuda_gpu_sum ./test
```

## ncu
