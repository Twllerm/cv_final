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

double uniform_transofrm(double value, double t, double vW) {
    double h = 2*(vW - t);

    double coeff;

    if (value < t - h/2) {
        coeff = 0;
    } else if (abs(value - t) < h/2) {
        coeff = ((value - t)/h) + 1/2;
    } else {
        coeff = 1;
    }

    return 255*coeff;
}

double normal_transofrm(double value, double t, double vW) {
    double sigma = (vW - t)/2.3263; // .99 quantile of  normal distribution

    return 255 - (255/2)*(1 - erf((value - t)/sqrt( 2*(sigma*sigma) )));
}

double fermi_dirac_transform(double value, double t, double vW) {
    double theta = (vW - t)/log(-1 + 1/0.99);

    return 255 - 255/(1 + exp(- ((value - t)/theta)) );
}

double binarize(double value, double threshold, double vW) {
    if (value >= threshold) {
        return 255;
    } else {
        return 0;
    }
}