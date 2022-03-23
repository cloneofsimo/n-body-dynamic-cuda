/*
Compile with nvcc -o main main.cu -I .. -lcuda $(pkg-config opencv4 --libs --cflags)

*/
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define N_BODY 3
#define MAX_T 2048
#define SAMPLE_RATE 4
#define DT 0.5

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

__global__ void traj_plot(float* out, float* n_bodys, int width, int height, float TENSION = 0.01, float DRAG = 0.001)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_n_bodys[N_BODY * 3];

    const int row_size = width * height;
    int idx = Row * width + Col;

    if (threadIdx.x < N_BODY) {
        s_n_bodys[threadIdx.x * 3 + 0] = n_bodys[threadIdx.x * 3 + 0];
        s_n_bodys[threadIdx.x * 3 + 1] = n_bodys[threadIdx.x * 3 + 1];
        s_n_bodys[threadIdx.x * 3 + 2] = n_bodys[threadIdx.x * 3 + 2];
    }

    __syncthreads();


    if (Row < height && Col < width) {
        float p_x = Col / (float)width * 4.0 - 2.0;
        float p_y = Row / (float)height * 4.0 - 2.0;

        float v_x = 0.0;
        float v_y = 0.0;


        for (int _t = 0; _t < MAX_T; _t++) {
            // update acceleration
            float ax = 0.0;
            float ay = 0.0;
            for (int i = 0; i < N_BODY; i++) {
                float dx = p_x - s_n_bodys[i * 3 + 0];
                float dy = p_y - s_n_bodys[i * 3 + 1];
                float this_weight = s_n_bodys[i * 3 + 2];
                float r2 = dx * dx + dy * dy;
                float r3 = r2 * sqrt(r2) + 1e-3;
                ax -= this_weight * dx / r3;
                ay -= this_weight * dy / r3;
            }



            float r = sqrt(p_x * p_x + p_y * p_y + 8);
            float dist = sqrt(p_x * p_x + p_y * p_y);

            float sintheta = dist / r;
            float cossqtheta = 1 - sintheta * sintheta;

            ax = ax * cossqtheta - TENSION * p_x - DRAG * v_x;
            ay = ay * cossqtheta - TENSION * p_y - DRAG * v_y;

            // update velocity
            v_x += ax * DT;
            v_y += ay * DT;

            // update position

            out[(_t * 2) * row_size + idx] = p_x;
            out[(_t * 2 + 1) * row_size + idx] = p_y;

            p_x += v_x * DT;
            p_y += v_y * DT;


        }

    }
}


void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <video output path> <weight> <string tension> <drag> \n", prog_name);
    exit(EXIT_FAILURE);
}


int main(int argc, char** argv)
{
    if (argc != 5) {
        Usage(argv[0]);
    }

    const char* file_name = argv[1];
    const float weight = atof(argv[2]);
    const float tension = atof(argv[3]);
    const float drag = atof(argv[4]);

    int width = 512, height = 512;


    float* d_result_dynamic; // [MAX_T][2][width * height];
    float* d_bodys; // x, y, weight

    gpuErrchk(cudaMallocManaged((void**)&d_result_dynamic, width * height * MAX_T * 2 * sizeof(float)));
    gpuErrchk(cudaMallocManaged((void**)&d_bodys, N_BODY * 3 * sizeof(float)));

    for (int i = 0; i < N_BODY; i++) {
        d_bodys[i * 3] = (float)sin(i * 2 * M_PI / N_BODY);
        d_bodys[i * 3 + 1] = (float)cos(i * 2 * M_PI / N_BODY);
        d_bodys[i * 3 + 2] = weight;
    }

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y));

    traj_plot << <grid, threads >> > (d_result_dynamic, d_bodys, width, height, tension, drag);

    gpuErrchk(cudaDeviceSynchronize());


    cv::Size frame_size(width, height);
    int frames_per_second = 60;
    cv::VideoWriter oVideoWriter(file_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        frames_per_second, frame_size, true);

    // randomly select pixels;
    std::vector<int> selected_pixels;
    for (int i = 0; i < width * height; i++) {
        if (rand() % 1000 == 0) {
            selected_pixels.push_back(i);
        }
    }

    for (int t = 0; t < MAX_T; t += SAMPLE_RATE) {
        cv::Mat frame(frame_size, CV_8UC3, cv::Scalar(0, 0, 0));
        // draw dynamics
        for (int pix_idx = 0; pix_idx < selected_pixels.size(); pix_idx++) {
            int idx = selected_pixels[pix_idx];

            float p_x = d_result_dynamic[(t * 2) * width * height + idx];
            float p_y = d_result_dynamic[(t * 2 + 1) * width * height + idx];
            if (pix_idx == 0) {
                std::cout << "p_x: " << p_x << " p_y: " << p_y << std::endl;
            }
            int x = (int)((p_x + 2.0) * width / 4.0);
            int y = (int)((p_y + 2.0) * height / 4.0);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                frame.at<cv::Vec3b>(y, x)[0] = 255;
                frame.at<cv::Vec3b>(y, x)[1] = 255;
                frame.at<cv::Vec3b>(y, x)[2] = 255;
            }
        }
        oVideoWriter.write(frame);

    }

    oVideoWriter.release();


    cudaFree(d_result_dynamic);
    cudaFree(d_bodys);

    return 0;
}