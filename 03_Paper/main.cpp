#include <cmath>
#include <core.hpp>
#include <core/hal/interface.h>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <iostream>
#include <opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int get_cdf(const Mat &src, Mat &cdf);
int get_segments(const Mat &src, const Mat &cdf, vector<Mat> &segments);
int transfer_color(const Mat &src, const Mat &src_segments, const Mat &target,
                   const Mat &target_segments, Mat &output);
float get_mean(const Mat &src);
float get_stdev(const Mat &src, float mean);
int apply_new_color(const vector<Mat> &src_channels,
                    const vector<Mat> &src_segments,
                    const vector<Mat> &target_channels,
                    const vector<Mat> &target_segments, Mat &output);

const string img_source = "./images/source2.jpg";
const string img_target = "./images/target.jpg";

int main(void) {
  // Input and target images reading
  Mat input = imread(img_source, IMREAD_COLOR);
  Mat target = imread(img_target, IMREAD_COLOR);
  if (!input.data or !input.data) {
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

  Mat input_cdf_a, input_cdf_b;
  Mat target_cdf_a, target_cdf_b;
  vector<Mat> input_segments_a, input_segments_b;
  vector<Mat> target_segments_a, target_segments_b;
  get_cdf(input_channels[1], input_cdf_a);
  get_cdf(input_channels[2], input_cdf_b);
  get_cdf(target_channels[1], target_cdf_a);
  get_cdf(target_channels[2], target_cdf_b);
  get_segments(input_channels[1], input_cdf_a, input_segments_a);
  get_segments(input_channels[2], input_cdf_b, input_segments_b);
  get_segments(target_channels[1], target_cdf_a, target_segments_a);
  get_segments(target_channels[2], target_cdf_b, target_segments_b);

  Mat ct_channel_a, ct_channel_b;
  apply_new_color(input_channels, input_segments_a, target_channels,
                  target_segments_a, ct_channel_a);
  apply_new_color(input_channels, input_segments_b, target_channels,
                  target_segments_b, ct_channel_b);

  Mat bgr_channel_a, bgr_channel_b;
  cvtColor(ct_channel_a, bgr_channel_a, COLOR_Lab2BGR);
  cvtColor(ct_channel_b, bgr_channel_b, COLOR_Lab2BGR);

  imshow("Color transfer channel A", bgr_channel_a);
  imshow("Color transfer channel B", bgr_channel_b);

  waitKey(0);
  return 0;
}

int get_cdf(const Mat &src, Mat &cdf) {
  int histSize = 256;       // number of bins in the histogram
  float range[] = {1, 257}; // range of pixel values
  const float *histRange = {range};

  // calculate the histogram
  Mat hist;
  calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

  // calculate the cumulative distribution function (CDF)
  hist.copyTo(cdf);
  for (int i = 1; i < histSize; i++)
    cdf.at<float>(i) += cdf.at<float>(i - 1);

  normalize(cdf, cdf, 0, 1, NORM_MINMAX);
  return 0;
}

int get_segments(const Mat &src, const Mat &cdf, vector<Mat> &segments) {
  int fq, med, tq;
  int segm[5] = {1, 0, 0, 0, 256};
  for (int i = 1; i < 256; i++) {
    float value_tmp = cdf.at<float>(i);
    if (value_tmp <= 0.25)
      segm[1] = i;
    if (value_tmp == segm[1] || value_tmp <= 0.50)
      segm[2] = i;
    if (value_tmp == segm[2] || value_tmp <= 0.75)
      segm[3] = i;
  }

  for (int i = 1; i < 5; i++) {
    Mat tmp = Mat::zeros(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
      uchar *p_dst = (uchar *)tmp.ptr<uchar>(r);
      uchar *p_src = (uchar *)src.ptr<uchar>(r);
      for (int c = 0; c < src.cols; c++) {
        if (p_src[c] >= segm[i - 1] && p_src[c] < segm[i])
          p_dst[c] = 1;
      }
    }
    segments.push_back(tmp);
  }
  return 0;
}

int transfer_color(const Mat &src, const Mat &src_segment, const Mat &target,
                   const Mat &target_segment, Mat &output) {
  Mat src_tmp = Mat::zeros(src.size(), CV_8UC1);
  Mat target_tmp = Mat::zeros(src.size(), CV_8UC1);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_msk = (uchar *)src_segment.ptr<uchar>(r);
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    uchar *p_dst = (uchar *)src_tmp.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      if (p_msk[c]) {
        p_dst[c] = p_src[c];
      }
    }
  }
  for (int r = 0; r < src.rows; r++) {
    uchar *p_msk = (uchar *)target_segment.ptr<uchar>(r);
    uchar *p_src = (uchar *)target.ptr<uchar>(r);
    uchar *p_dst = (uchar *)target_tmp.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      if (p_msk[c]) {
        p_dst[c] = p_src[c];
      }
    }
  }
  imshow("Imagen yo:", src_tmp);
  waitKey(0);

  src_tmp.copyTo(output);

  float src_mean = get_mean(src_tmp);
  float src_stdev = get_stdev(src_tmp, src_mean);
  float target_mean = get_mean(target_tmp);
  float target_stdev = get_stdev(target_tmp, target_mean);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src_tmp.ptr<uchar>(r);
    uchar *p_dst = (uchar *)output.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      p_dst[c] = (target_stdev / src_stdev);
      p_dst[c] *= p_src[c] - src_mean;
      p_dst[c] += target_mean;
    }
  }
  return 0;
}

float get_mean(const Mat &src) {
  float mean = 0.0;
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      mean += p_src[c];
    }
  }
  return mean * (1.0 / (src.rows * src.cols));
}

float get_stdev(const Mat &src, float mean) {
  float stdev = 0.0;
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      stdev += (p_src[c] - mean) * (p_src[c] - mean);
    }
  }
  stdev *= 1.0 / (src.rows * src.cols);
  return sqrt(stdev);
}

int apply_new_color(const vector<Mat> &src_channels,
                    const vector<Mat> &src_segments,
                    const vector<Mat> &target_channels,
                    const vector<Mat> &target_segments, Mat &output) {
  vector<vector<Mat>> tmp_out;
  for (int i = 0; i < 4; i++) {
    vector<Mat> tmp_segments;
    for (int j = 0; j < 3; j++) {
      Mat tmp = Mat::zeros(src_channels[0].size(), CV_8UC1);
      transfer_color(src_channels[j], src_segments[i], target_channels[j],
                     target_segments[i], tmp);
      tmp_segments.push_back(tmp);
    }
    tmp_out.push_back(tmp_segments);
  }

  vector<Mat> segments_lab;
  for (int i = 0; i < 4; i++) {
    Mat tmp_array[3] = {tmp_out[i][0], tmp_out[i][1], tmp_out[i][2]};
    Mat tmp = Mat::zeros(src_channels[0].size(), CV_8UC3);
    merge(tmp_array, 3, tmp);
    segments_lab.push_back(tmp);
  }

  for (int i = 0; i < 4; i++) {
    Mat tmp;
    bitwise_or(segments_lab[0], segments_lab[1], output);
    bitwise_or(output, segments_lab[2], output);
    bitwise_or(output, segments_lab[3], output);
  }

  return 0;
}
