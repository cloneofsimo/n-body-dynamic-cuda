/*
Compile with nvcc -o main main.cu -I .. -lcuda $(pkg-config opencv4 --libs --cflags)

*/
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define N_BODY_MAX 16
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

__global__ void traj_plot(float* out, float* n_bodys_data, int width, int height, float TENSION = 0.01, float DRAG = 0.001, int N_BODY = 3)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_n_bodys_data[N_BODY_MAX * 3];

    const int row_size = width * height;
    int idx = Row * width + Col;

    if (threadIdx.x < N_BODY) {
        s_n_bodys_data[threadIdx.x * 3 + 0] = n_bodys_data[threadIdx.x * 3 + 0];
        s_n_bodys_data[threadIdx.x * 3 + 1] = n_bodys_data[threadIdx.x * 3 + 1];
        s_n_bodys_data[threadIdx.x * 3 + 2] = n_bodys_data[threadIdx.x * 3 + 2];
    }

    __syncthreads();


    if (Row < height && Col < width) {
        float p_x = Col / (float)width * 4.0 - 2.0 + 1e-6;
        float p_y = Row / (float)height * 4.0 - 2.0 + 1e-6;

        float v_x = 0.0;
        float v_y = 0.0;


        for (int _t = 0; _t < MAX_T; _t++) {
            // update acceleration
            float ax = 0.0;
            float ay = 0.0;
            for (int i = 0; i < N_BODY; i++) {
                float dx = p_x - s_n_bodys_data[i * 3 + 0];
                float dy = p_y - s_n_bodys_data[i * 3 + 1];
                float this_weight = s_n_bodys_data[i * 3 + 2];
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
    fprintf(stderr, "Usage: %s <video output path> <weight> <string tension> <drag> <n_body> \n", prog_name);
    exit(EXIT_FAILURE);
}

// return nth bgr rainbow color with n colors, using hsv color space
cv::Scalar RainbowColor(int n, int n_body) {
    int h = (int)round((180.0 * n) / ((float)n_body + 1));
    std::cout << h << std::endl;
    cv::Mat hsv = cv::Mat(1, 1, CV_8UC3, cv::Scalar(h, 255, 255));
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);
}


int main(int argc, char** argv)
{
    if (argc != 6) {
        Usage(argv[0]);
    }

    const char* file_name = argv[1];
    const float weight = atof(argv[2]);
    const float tension = atof(argv[3]);
    const float drag = atof(argv[4]);
    const float n_body = atoi(argv[5]);

    int width = 512, height = 512;


    float* d_result_dynamic; // [MAX_T][2][width * height];
    float* d_bodys; // x, y, weight

    gpuErrchk(cudaMallocManaged((void**)&d_result_dynamic, width * height * MAX_T * 2 * sizeof(float)));
    gpuErrchk(cudaMallocManaged((void**)&d_bodys, n_body * 3 * sizeof(float)));

    for (int i = 0; i < n_body; i++) {
        d_bodys[i * 3] = (float)sin(i * 2 * M_PI / n_body);
        d_bodys[i * 3 + 1] = (float)cos(i * 2 * M_PI / n_body);
        d_bodys[i * 3 + 2] = weight;
    }

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y));

    traj_plot << <grid, threads >> > (d_result_dynamic, d_bodys, width, height, tension, drag, n_body);

    gpuErrchk(cudaDeviceSynchronize());


    cv::Size frame_size(width, height);
    int frames_per_second = 60;
    cv::VideoWriter oVideoWriter(file_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        frames_per_second, frame_size, true);

    // randomly select pixels;
    std::vector<int> selected_pixels, final_color;

    for (int i = 0; i < width * height; i++) {
        if (rand() % 1 == 0) {
            selected_pixels.push_back(i);
        }
    }
    const int FINAL_T = MAX_T - 1;
    for (int pix : selected_pixels) {
        float p_x = d_result_dynamic[(FINAL_T * 2) * width * height + pix];
        float p_y = d_result_dynamic[(FINAL_T * 2 + 1) * width * height + pix];

        int x = (int)(p_x + 2.0) * width / 4;
        int y = (int)(p_y + 2.0) * height / 4;

        if (x < 0 || x >= width || y < 0 || y >= height) {
            final_color.push_back(-1);
            continue;
        }

        int min_dist_idx = -1;
        float min_dist = 1e10;
        for (int i = 0; i < n_body; i++) {
            float dx = p_x - d_bodys[i * 3];
            float dy = p_y - d_bodys[i * 3 + 1];
            float dist = sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
                min_dist_idx = i;
            }
        }

        final_color.push_back(min_dist_idx);

    }
    std::cout << "selected pixels: " << selected_pixels.size() << std::endl;

    // rainbow colormap
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < n_body + 1; i++) {
        colors.push_back(RainbowColor(i, n_body));
    }
    for (int i = 0; i < n_body + 1; i++) {
        std::cout << i << ": " << colors[i] << std::endl;
    }




    for (int t = 0; t < MAX_T; t += SAMPLE_RATE) {
        cv::Mat frame(frame_size, CV_8UC3, cv::Scalar(0, 0, 0));
        // draw dynamics
        for (int pix_idx = 0; pix_idx < selected_pixels.size(); pix_idx++) {
            int idx = selected_pixels[pix_idx];
            int color_idx = final_color[pix_idx];

            float p_x = d_result_dynamic[(t * 2) * width * height + idx];
            float p_y = d_result_dynamic[(t * 2 + 1) * width * height + idx];
            if (pix_idx == 0) {
                std::cout << "p_x: " << p_x << " p_y: " << p_y << std::endl;
            }
            int x = (int)((p_x + 2.0) * width / 4.0);
            int y = (int)((p_y + 2.0) * height / 4.0);

            if (x >= 0 && x < width && y >= 0 && y < height) {

                cv::circle(frame, cv::Point(x, y), 1, colors[color_idx + 1], -1);

            }
        }
        oVideoWriter.write(frame);

    }

    oVideoWriter.release();


    cudaFree(d_result_dynamic);
    cudaFree(d_bodys);

    return 0;
}