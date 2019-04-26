#include <iostream>
#include <math.h> 
#include <map>
#include <numeric> 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "shade_substraction.cpp"
#include "grayscale_transformations.cpp"
#include <iostream>

using std::cin;
using std::cout;
using std::vector;
using std::pair;
using std::make_pair;
using std::map;
using std::endl;
using namespace cv;

int thr = 150; 
int k = 10; 
Mat image;
Mat image_substracted ;


void drawHist(const vector<int>& data, Mat& dst, int binSize, int height, Scalar color1, Scalar color2)
{
    int max_value = *max_element(data.begin(), data.end());
    int rows = height + 20;
    int cols = 0;
    
    cols = data.size() * binSize;
    dst = Mat3b(rows, cols, Vec3b(0,0,0));

    for (int i = 0; i < data.size(); ++i)
    {
       int h = height -  (height*data[i])/max_value;
       rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), (i%2) ? color1 : color2, CV_FILLED);
    }

}

double evaluate_vW(Mat image) {
    double result = 0;

    map<int, int> vCounts;

     for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < image.channels();  c++) {
                double value = image.at<Vec3b>(y,x)[c];

                auto currentCount = vCounts.find(value);

                if(currentCount == vCounts.end()) {
                     vCounts.insert(make_pair(value, 0));
                } else {
                    currentCount->second += 1;
                }

            }
        }
    }

    double numerator = 0;
    double denumerator = 0;

    for(auto const& [key, count] : vCounts)
    {
        numerator += key * count;
        denumerator += count;
    }

    return numerator/denumerator;
       
}

void show_result(Mat image, String title, double threshold, double vW, double (*tranform_func)(double, double, double)) {
    Mat new_image = Mat::zeros(image.size(), image.type());

    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            for(int c = 0; c < image.channels();  c++) {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>((*tranform_func)(image.at<Vec3b>(y,x)[c], threshold, vW));
            }
        }
    }

    Mat result;

    hconcat(image, new_image, result);

    imshow(title, new_image);
}

void on_trackbar(int, void*) {

    double vW = evaluate_vW(image);

    show_result(image, "Uniform transform", thr, vW, uniform_transofrm);
    show_result(image, "Normal transofrm", thr, vW, normal_transofrm);
    show_result(image, "Logistic transofrm", thr, vW, fermi_dirac_transform);
    show_result(image, "Simple threshold", thr, vW, binarize);

    double vW_substracted = evaluate_vW(image_substracted);
    show_result(image_substracted, "Local thresholding via shade substraction", thr, vW_substracted, normal_transofrm);

}

void on_trackbar_k(int, void*) {
    image_substracted = substract_shades(image, k);

    double vW_substracted = evaluate_vW(image_substracted);
    show_result(image_substracted, "Local thresholding via shade substraction", thr, vW_substracted, normal_transofrm);
}


void create_GUI() {
    namedWindow("Parameters", 1);

    imshow("Original", image);

    createTrackbar("threshold", "Parameters", &thr, 255, on_trackbar);
    createTrackbar("k for shade substraction", "Parameters", &k, 100, on_trackbar_k);
}


int main(int argc, char** argv)
{

    Mat raw = imread("./data/test.jpg");
    resize(raw, image, Size(200, 200), 0, 0, INTER_CUBIC);
    
    void *ptr;
    on_trackbar_k(1, ptr);

    if(image.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
      return -1;
    }

    create_GUI();

    waitKey();
    waitKey();
    return 0;
}