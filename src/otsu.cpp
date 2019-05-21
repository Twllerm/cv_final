#include <iostream>
#include <math.h> 
#include <map>
#include <numeric> 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// Количество уровней интенсивности у изображения.
// Стандартно для серого изображения -- это 256. От 0 до 255.
const int INTENSITY_LAYER_NUMBER = 256;

// Возвращает гистограмму по интенсивности изображения от 0 до 255 включительно
vector <int> calculateHist(Mat image) {

    vector<int> hist;
    
    // Иницилизируем гистограмму нулями
    for (int i = 0; i < INTENSITY_LAYER_NUMBER; ++i) {
        hist.push_back(0);
    }

    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            ++hist[image.at<Vec3b>(y,x)[0]];
        }
    }

    return hist;
}

// Посчитать сумму всех интенсивностей
int calculateIntensitySum(Mat image) {
    int sum = 0;

     for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            sum += image.at<Vec3b>(y,x)[0];
        }
    }

    return sum;
}

// Функция возвращает порог бинаризации для полутонового изображения image с общим числом пикселей size.
// const IMAGE *image содержит интенсивность изображения от 0 до 255 включительно.
// size -- количество пикселей в image.

int otsuThreshold(Mat image) {
    vector <int> hist = calculateHist(image);

    // Необходимы для быстрого пересчета разности дисперсий между классами
    int all_pixel_count = image.cols * image.rows;
    int all_intensity_sum = calculateIntensitySum(image);

    int best_thresh = 0;
    double best_sigma = 0.0;

    int first_class_pixel_count = 0;
    int first_class_intensity_sum = 0;

    // Перебираем границу между классами
    // thresh < INTENSITY_LAYER_NUMBER - 1, т.к. при 255 в ноль уходит знаменатель внутри for
    for (int thresh = 0; thresh < INTENSITY_LAYER_NUMBER - 1; ++thresh) {
        first_class_pixel_count += hist[thresh];
        first_class_intensity_sum += thresh * hist[thresh];

        double first_class_prob = first_class_pixel_count / (double) all_pixel_count;
        double second_class_prob = 1.0 - first_class_prob;

        double first_class_mean = first_class_intensity_sum / (double) first_class_pixel_count;
        double second_class_mean = (all_intensity_sum - first_class_intensity_sum) 
            / (double) (all_pixel_count - first_class_pixel_count);

        double mean_delta = first_class_mean - second_class_mean;

        double sigma = first_class_prob * second_class_prob * mean_delta * mean_delta;

        if (sigma > best_sigma) {
            best_sigma = sigma;
            best_thresh = thresh;
        }
    }

   return best_thresh;
}

void drawHist(const vector<int>& data, Mat& dst)
{
    int binSize = 3;
    int height = 100;
    int max_value = *max_element(data.begin(), data.end());
    int rows = height + 20;
    int cols = 0;

    Scalar color = Scalar(255, 255, 255);
    
    cols = data.size() * binSize;
    dst = Mat3b(rows, cols, Vec3b(0,0,0));

    for (int i = 0; i < data.size(); ++i)
    {
       int h = height -  (height*data[i])/max_value;
       rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), color, CV_FILLED);
    }

}

Mat plotGraph(vector<int>  vals, int YRange[2])
{

    auto it = minmax_element(vals.begin(), vals.end());
    float scale = 1./ceil(*it.second - *it.first); 
    float bias = *it.first;
    int rows = YRange[1] - YRange[0] + 1;
    cv::Mat image = Mat::zeros( rows, vals.size(), CV_8UC3 );
    image.setTo(0);
    for (int i = 0; i < (int)vals.size()-1; i++)
    {
        cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i+1, rows - 1 - (vals[i+1] - bias)*scale*YRange[1]), Scalar(255, 255, 255), 1);
    }

    return image;
}