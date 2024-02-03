#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

int main() {
    for (int i = 1; i <= 10; i++) {
        cv::Mat image = cv::imread("../giena1024x768.jpg");

        setlocale(LC_ALL, "Russian");
        if (image.empty()) {
            std::cout << "Не удалось загрузить изображение." << std::endl;
            return -1;
        }

        // Создание двух пустых Mat-объектов для модифицированных синего и желтого каналов
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat modifiedBlueChannel = cv::Mat::zeros(image.size(), CV_8U);
        cv::Mat modifiedYellowChannel = cv::Mat::zeros(image.size(), CV_8U);

#pragma omp parallel for
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                uchar redv = pixel[2];
                uchar greenv = pixel[1];
                uchar bluev = pixel[0];

                // Вычисление модифицированного синего канала
                uchar Bv = bluev - (greenv + bluev) / 2;

                // Вычисление модифицированного желтого канала
                uchar Yv = redv + greenv - 2 * (std::abs(redv - greenv) + bluev);

                // Запись результатов в соответствующие Mat-объекты
                modifiedBlueChannel.at<uchar>(y, x) = Bv;
                modifiedYellowChannel.at<uchar>(y, x) = Yv;
            }
        }


        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << i << ". Time " << diff.count() << " ms" << std::endl;
        // Сохранение изображений в файлы
        cv::imwrite("../modified_blue_channel.jpg", modifiedBlueChannel);
        cv::imwrite("../modified_yellow_channel.jpg", modifiedYellowChannel);
    }
    return 0;
}
