#include <iostream>
#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>

using namespace std;
using namespace cv;

int claheImage(const Mat &src, Mat &dst);
int getAdaptiveThreshold(const Mat &src, Mat &dst, int thresholdType);
int getKMeans(const Mat &src, Mat &dst, int clusters, int iterations);

int main(void)
{
    string imagepath = "./images/test.png";
    string maskpath = "./images/real_mask.png";
    Mat input = imread(imagepath, IMREAD_GRAYSCALE);
    if(!input.data)
    {
        cout << "Wrong or non existent filename" << endl;
        return -1;
    }
    imshow("Input image", input);
    Mat realMask = imread(maskpath, IMREAD_GRAYSCALE);
    if(!realMask.data)
    {
        cout << "Wrong or non existent filename" << endl;
        return -1;
    }
    imshow("Real mask", realMask);

    Mat clahe;
    claheImage(input, clahe);
    imshow("Clahe image", clahe);

    Mat adThresholdMean;
    getAdaptiveThreshold(input, adThresholdMean, ADAPTIVE_THRESH_MEAN_C);
    imshow("Adaptive threshold without clahe, MEAN", adThresholdMean);

    Mat adThresholdGaussian;
    getAdaptiveThreshold(input, adThresholdGaussian, ADAPTIVE_THRESH_GAUSSIAN_C);
    imshow("Adaptive threshold without clahe, GAUSSIAN", adThresholdGaussian);

    Mat adThresholdMeanClahe;
    getAdaptiveThreshold(clahe, adThresholdMeanClahe, ADAPTIVE_THRESH_MEAN_C);
    imshow("Adaptive threshold with clahe, MEAN", adThresholdMeanClahe);

    Mat adThresholdGaussianClahe;
    getAdaptiveThreshold(clahe, adThresholdGaussianClahe, ADAPTIVE_THRESH_GAUSSIAN_C);
    imshow("Adaptive threshold with clahe, GAUSSIAN", adThresholdGaussianClahe);

    Mat otsu;
    threshold(input, otsu, 0, 255, THRESH_OTSU);
    imshow("Otsu threshold without clahe", otsu);
    
    Mat otsuClahe;
    threshold(clahe, otsuClahe, 0, 255, THRESH_OTSU);
    imshow("Otsu threshold from clahe", otsuClahe);
    
    Mat segmentedImage;
    getKMeans(input, segmentedImage, 5, 5);
    imshow("Kmeans without clahe", segmentedImage);
    
    Mat segmentedImageClahe;
    getKMeans(clahe, segmentedImageClahe, 5, 5);
    imshow("Kmeans with clahe", segmentedImageClahe);
    
    Mat otsuKMeans;
    threshold(segmentedImage, otsuKMeans, 0, 255, THRESH_OTSU);
    imshow("Otsu threshold from K-Means", otsuKMeans);
    
    Mat otsuKMeansClahe;
    threshold(segmentedImageClahe, otsuKMeansClahe, 0, 255, THRESH_OTSU);
    imshow("Otsu threshold from K-Means with clahe", otsuKMeansClahe);

    waitKey(0);
    return 0;
}

int claheImage(const Mat &src, Mat &dst)
{
    Ptr<CLAHE> cobj = createCLAHE(2.0, Size(11,11));
    cobj->apply(src, dst);
    return 0;
}

int getAdaptiveThreshold(const Mat &src, Mat &dst, int thresholdType)
{
    Mat blur;
    if (thresholdType == ADAPTIVE_THRESH_MEAN_C)
        GaussianBlur(src, blur, Size(5, 5), 3, 3);
    else
        medianBlur(src, blur, 5);
    adaptiveThreshold(src, dst, 255, thresholdType, THRESH_BINARY_INV, 11, 2);
    return 0;
}

int getKMeans(const Mat &src, Mat &dst, int clusters, int iterations)
{
    int sizeline = src.rows * src.cols;
    Mat data = src.reshape(1, sizeline);
    data.convertTo(data, CV_32F);
    vector<int> labels;
    Mat1f colors;
    kmeans(data, 3,labels,TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.),3,KMEANS_PP_CENTERS,colors);
    for (int i = 0; i < sizeline; i++)
    {
        float *pixel = data.ptr<float>(i);
        pixel[0] = colors(labels[i], 0);
        pixel[1] = colors(labels[i], 1);
        pixel[2] = colors(labels[i], 2);
    }
    dst = data.reshape(1, src.rows);
    dst.convertTo(dst, CV_8U);
    return 0;
}
