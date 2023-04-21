#include <iostream>
#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

int cumulative_histo(const Mat &src, Mat &cdf, Mat &thresholded, int &fq, int &med, int &tq);

const string img_source = "./images/source5.jpg";
const string img_target = "./images/target3.jpg";

int main(void)
{
    // Input and target images reading
    Mat input = imread(img_source, IMREAD_COLOR);
    Mat target = imread(img_target, IMREAD_COLOR);
    if (!input.data or !input.data)
    {
        cout << "Wrong or non existent input or target filename" << endl;
        return -1;
    }
    imshow("Input image", input);
    imshow("Target image", target);

    // From BGR to CIELAB
    Mat input_cielab;
    Mat target_cielab;
    cvtColor(input, input_cielab, COLOR_BGR2Lab);
    cvtColor(target, target_cielab, COLOR_BGR2Lab);

    // Split channels
    vector<Mat> input_channels;
    vector<Mat> target_channels;
    split(input_cielab, input_channels);
    split(target_cielab, target_channels);

    // Multilevel thresholding channel A
    int fq, med, tq;
    Mat histo_channel_a_input, thresholded_a_input, thresholded_a_input_channels;
    cumulative_histo(input_channels[1], histo_channel_a_input, thresholded_a_input, fq, med, tq);
//    imshow("Multilevel Thresholding - Channel A - Input", thresholded_a_input);

    Mat histo_channel_a_target, thresholded_a_target, thresholded_a_target_channels;
    cumulative_histo(target_channels[1], histo_channel_a_target, thresholded_a_target, fq, med, tq);
//    imshow("Multilevel Thresholding - Channel A - Target", thresholded_a_target);

    // Multilevel thresholding channel B
    Mat histo_channel_b_input, thresholded_b_input, thresholded_b_input_channels;
    cumulative_histo(input_channels[2], histo_channel_b_input, thresholded_b_input, fq, med, tq);
//    imshow("Multilevel Thresholding - Channel B - Input", thresholded_b_input);

    Mat histo_channel_b_target, thresholded_b_target, thresholded_b_target_channels;
    cumulative_histo(target_channels[2], histo_channel_b_target, thresholded_b_target, fq, med, tq);
//    imshow("Multilevel Thresholding - Channel B - Target", thresholded_b_target);

    // Multilevel Color Transfer channel A
    Mat mean_input, std_dev_input;
    Mat mean_target, std_dev_target;
    Mat output = Mat::zeros(input.size(), CV_8UC3);
    vector<Mat> thresholded_a_input_channels_s;
    vector<Mat> thresholded_a_target_channels_s;
    split(thresholded_a_input, thresholded_a_input_channels_s);
    split(thresholded_a_target, thresholded_a_target_channels_s);

    // Four levels of Threshold
//    Mat color_transfer = Mat::zeros(input.size(), CV_8UC1);
    for (int i=0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Mat tmpInput = Mat::zeros(input.size(), CV_8UC1);
            Mat tmpTarget = Mat::zeros(target.size(), CV_8UC1);
            float meanInput = 0;
            float meanTarget = 0;
            float stdevInput = 0;
            float stdevTarget = 0;
            for (int r = 0; r < input.rows; r++)
            {
                uchar *pThresh = (uchar *)thresholded_a_input_channels_s[i].ptr<uchar>(r);
                uchar *pDst = (uchar *)tmpInput.ptr<uchar>(r);
                Vec3b *pIn = input_cielab.ptr<Vec3b>(r);
                for (int c = 0; c < input.cols; c++)
                {
                    if (pThresh[c])
                    {
                        pDst[c] = pIn[c][j];
                        meanInput += pIn[c][j];
                    }
                }
            }
            for (int r = 0; r < target.rows; r++)
            {
                uchar *pThresh = (uchar *)thresholded_a_target_channels_s[i].ptr<uchar>(r);
                uchar *pDst = (uchar *)tmpTarget.ptr<uchar>(r);
                Vec3b *pIn = target_cielab.ptr<Vec3b>(r);
                for (int c = 0; c < target.cols; c++)
                {
                    if (pThresh[c])
                    {
                        pDst[c] = pIn[c][j];
                        meanTarget += pIn[c][j];
                    }
                }
            }
            meanInput = (1 / input_cielab.rows * input_cielab.cols) * meanInput;
            meanTarget = (1 / target_cielab.rows * target_cielab.cols) * meanTarget;

            for (int r = 0; r < input.rows; r++)
            {
                uchar *pThresh = (uchar *)thresholded_a_input_channels_s[i].ptr<uchar>(r);
                Vec3b *pIn = input_cielab.ptr<Vec3b>(r);
                for (int c = 0; c < input.cols; c++)
                {
                    if (pThresh[c])
                    {
                        float tmp = pIn[c][j] - meanInput;
                        stdevInput += (tmp * tmp);
                    }
                }
            }
            for (int r = 0; r < target.rows; r++)
            {
                uchar *pThresh = (uchar *)thresholded_a_target_channels_s[i].ptr<uchar>(r);
                Vec3b *pIn = target_cielab.ptr<Vec3b>(r);
                for (int c = 0; c < target.cols; c++)
                {
                    if (pThresh[c])
                    {
                        float tmp = pIn[c][j];
                        stdevTarget += (tmp * tmp);
                    }
                }
            }

            stdevInput = sqrt(stdevInput);
            stdevTarget = sqrt(stdevTarget);

            imshow("tmpIn", tmpInput);
            imshow("tmpTarg", tmpTarget);
            waitKey(0);

            meanStdDev(tmpInput, mean_input, std_dev_input);
            meanStdDev(tmpTarget, mean_target, std_dev_target);
            for (int r = 0; r < output.rows; r++)
            {
                Vec3b *pDst = output.ptr<Vec3b>(r);
                Vec3b *pChannels = input_cielab.ptr<Vec3b>(r);
                uchar *pThresh = (uchar *)thresholded_a_input_channels_s[i].ptr<uchar>(r);
                for (int c = 0; c < output.cols; c++)
                {
//                    if (pThresh[c] == 1)
//                    {
                        float tmp = ((stdevTarget/stdevInput)*(pChannels[c][j] - meanInput) + meanTarget);
                        pDst[c][j] = (uchar)tmp;
//                    }
                }
            }
            Mat tmp_out;
            cvtColor(output, tmp_out, COLOR_Lab2BGR);
            imshow("tmpOut", tmp_out);
            waitKey(0);
        }
    }
    Mat output_bgr;
    cvtColor(output, output_bgr, COLOR_Lab2BGR);
    imshow("Output image - Thresholded Channel A", output_bgr);

    // Multilevel Color Transfer channel A


    // Merging channels
//    Mat thresholded_channels_input_a[3] = {input_channels[0], color_transfer, input_channels[2]};
//    Mat thresholded_channels_input_b[3] = {input_channels[0], input_channels[1], thresholded_b_input};
//    Mat thresholded_channels_target_a[3] = {target_channels[0], thresholded_a_target, target_channels[2]};
//    Mat thresholded_channels_target_b[3] = {target_channels[0], target_channels[1], thresholded_b_target};

    Mat thresholded_merged_input_a, thresholded_merged_input_b;
    Mat thresholded_merged_target_a, thresholded_merged_target_b;
//    merge(thresholded_channels_input_a, 3, thresholded_merged_input_a);
//    merge(thresholded_channels_input_b, 3, thresholded_merged_input_b);
//    merge(thresholded_channels_target_a, 3, thresholded_merged_target_a);
//    merge(thresholded_channels_target_b, 3, thresholded_merged_target_b);

    // From CIELAB to BGR
    Mat input_bgr_a, input_bgr_b;
    Mat target_bgr_a, target_bgr_b;
//    cvtColor(thresholded_merged_input_a, input_bgr_a, COLOR_Lab2BGR);
//    cvtColor(thresholded_merged_input_b, input_bgr_b, COLOR_Lab2BGR);
//    cvtColor(thresholded_merged_target_a, target_bgr_a, COLOR_Lab2BGR);
//    cvtColor(thresholded_merged_target_b, target_bgr_b, COLOR_Lab2BGR);

//    imshow("Input image - Thresholded Channel A", input_bgr_a);
//    imshow("Input image - Thresholded Channel B", input_bgr_b);
//    imshow("Target image - Thresholded Channel A", target_bgr_a);
//    imshow("Target image - Thresholded Channel B", target_bgr_b);

    // Histogram comparison channel A


    // Histogram comparison channel B


    // Edge detection


    // Canvas creation


    // Image fusion


    // Filtering Mean - Median


    // Show output image

    waitKey(0);
    return 0;
}

int cumulative_histo(const Mat &src, Mat &cdf, Mat &thresholded, int &fq, int &med, int &tq)
{
    int histSize = 256; // number of bins in the histogram
    float range[] = {1, 257}; // range of pixel values
    const float* histRange = {range};

    // calculate the histogram
    Mat hist;
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

    // calculate the cumulative distribution function (CDF)
    hist.copyTo(cdf);
    for (int i = 1; i < histSize; i++) cdf.at<float>(i) += cdf.at<float>(i-1);

    // normalize the CDF
    normalize(cdf, cdf, 0, 1, NORM_MINMAX);

    for (int i = 1; i < histSize; i++)
    {
        if (cdf.at<float>(i) < 0.25) fq = i;
        if (cdf.at<float>(i) < 0.50 || med == fq) med = i;
        if (cdf.at<float>(i) < 0.75 || tq == med) tq = i;
    }

    // apply the thresholds to the image
    thresholded = Mat::zeros(src.size(), CV_8UC4);
    for (int r = 0; r < thresholded.rows; r++)
    {
        Vec4b *pDst = thresholded.ptr<Vec4b>(r);
        uchar *pMsk = (uchar *)src.ptr<uchar>(r);
        for (int c = 0; c < thresholded.cols; c++)
        {
            if (pMsk[c] < fq) pDst[c][0] = 1;
            else if (pMsk[c] < med && pMsk[c] > fq) pDst[c][1] = 1;
            else if (pMsk[c] < tq && pMsk[c] > med) pDst[c][2] = 1;
            else pDst[c][3] = 1;
        }
    }
    return 0;
}
