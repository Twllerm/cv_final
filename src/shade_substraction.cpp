#include <iostream>
#include <math.h> 
#include <map>
#include <numeric> 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using std::cin;
using std::cout;
using std::vector;
using std::pair;
using std::make_pair;
using std::map;
using std::endl;
using namespace cv;

double eval_shade(Mat image, int y, int x, int c, int k) {
    int left = max(0, x - k);
    int right = min(x + k, image.cols);
    int top = max(0, y - k);
    int bottom = min(y + k, image.rows);

    double max_val = image.at<Vec3b>(y,x)[c];

    for (int yy = top; yy <= bottom; ++yy) {
         for (int xx = left; xx <= right; ++xx) {
            double val = image.at<Vec3b>(yy,xx)[c];

            if(val < max_val) {
                max_val = val;
            }
        }
    }

    return max_val;
}

Mat substract_shades(Mat image, int k) {
    Mat substractedImage = Mat::zeros(image.size(), image.type());

      for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            for(int c = 0; c < image.channels();  c++) {
                double udpated_value = image.at<Vec3b>(y,x)[c] - eval_shade(image, y, x, c, k);
                substractedImage.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(udpated_value);
            }
        }
    }

    return substractedImage;
}

Mat substract_shades_2(Mat image, int k) {
    Mat substractedImage = Mat::zeros(image.size(), image.type());

    int x_ticks = ceil(image.cols / k);
    int y_ticks = ceil(image.rows / k);

      for(int y_tick = 0; y_tick < image.rows; y_tick += k) {
        for(int x_tick = 0; x_tick < image.cols; x_tick +=k) {

            int left = max(0, x_tick - k);
            int right = min(x_tick+ k, image.cols);
            int top = max(0, y_tick - k);
            int bottom = min(y_tick + k, image.rows);

            double max_val = image.at<Vec3b>(top,left)[0];

            for(int y = top; y < bottom; y += 1) {
                for(int x = left; x < right; ++x) {
                    for(int c = 0; c < image.channels();  c++) {
                         double val = image.at<Vec3b>(y,x)[c];
                        if(val < max_val) {
                            max_val = val;
                        }
                    }
                }
            }

            for(int y = top; y < bottom; y += 1) {
                for(int x = left; x < right; ++x) {
                    for(int c = 0; c < image.channels();  c++) {
                        double udpated_value = image.at<Vec3b>(y,x)[c] + max_val;
                        substractedImage.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(udpated_value);
                    }
                }
            }

        }
    }

    return substractedImage;
}

