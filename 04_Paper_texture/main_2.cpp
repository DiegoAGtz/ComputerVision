#include <cmath>
#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <iostream>
#include <opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int get_cdf(const Mat &src, Mat &cdf);
int show_histo(Mat &hist, int histSize);
int get_segments(const Mat &src, const Mat &cdf, vector<Mat> &segments);
int color_transfer(const Mat &src, const Mat &src_segments, const Mat &target,
                   const Mat &target_segments, Mat &output);
float get_mean(const Mat &src);
float get_stdev(const Mat &src, float mean);
int apply_new_color(const vector<Mat> &src_channels,
                    const vector<Mat> &src_segments,
                    const vector<Mat> &target_channels,
                    const vector<Mat> &target_segments, Mat &output);
int get_real_segment(const Mat &src, const Mat &msk, Mat &real_segment);
float get_euclidean_distance(const Mat &src, const Mat &target);
int claheImage(const Mat &src, Mat &dst);
int get_edges(const Mat &src, Mat &edges);

const string img_source = "./images/source6.jpg";
const string img_target = "./images/target5.jpg";
const string img_texture = "./images/texture.jpg";

int main(void) {
  // Input and target images reading
  Mat input = imread(img_source, IMREAD_COLOR);
  Mat target = imread(img_target, IMREAD_COLOR);
  Mat texture = imread(img_texture, IMREAD_COLOR);
  if (!input.data || !target.data || !texture.data) {
    cout << "Wrong or non existent input or target filename" << endl;
    return -1;
  }
  imshow("Input image", input);
  imshow("Target image", target);

  // Apply clahe
  vector<Mat> input_bgr;
  vector<Mat> target_bgr;
  split(input, input_bgr);
  split(target, target_bgr);

  for (int i = 0; i < 3; i++) {
    claheImage(input_bgr[i], input_bgr[i]);
    claheImage(target_bgr[i], target_bgr[i]);
  }

  Mat input_clahe[] = {input_bgr[0], input_bgr[1], input_bgr[2]};
  Mat target_clahe[] = {target_bgr[0], target_bgr[1], target_bgr[2]};
  // merge(input_clahe, 3, input);
  // merge(target_clahe, 3, target);

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

  // for (int i = 0; i < 4; i++) {
  //   imshow("Segment A - " + to_string(i), input_segments_a[i]);
  //   imshow("Segment B - " + to_string(i), input_segments_b[i]);
  // }
  // waitKey(0);

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

  float eud_a, eud_b;
  eud_a = get_euclidean_distance(bgr_channel_a, target);
  eud_b = get_euclidean_distance(bgr_channel_b, target);

  cout << eud_a << endl;
  cout << eud_b << endl;

  Mat output, edges;
  if (eud_a < eud_b)
    bgr_channel_a.copyTo(output);
  else
    bgr_channel_b.copyTo(output);

  get_edges(input, edges);

  // Image with edges
  bitwise_or(output, edges, output);

  // Add texture
  addWeighted(output, 0.85, texture, 0.15, 0.0, output);
  blur(output, output, Size(3, 3));
  // medianBlur(output, output, 5);
  imshow("Output", output);

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

int show_histo(Mat &hist, int histSize) {
  // crear la trama del histograma
  int width = 512, height = 400;
  Mat histPlot(height, width, CV_8UC3, Scalar(0, 0, 0));

  // encontrar el valor máximo del histograma para la normalización
  double maxVal = 0;
  minMaxLoc(hist, 0, &maxVal);

  // dibujar el histograma
  int binWidth = cvRound((double)width / histSize);
  for (int i = 0; i < histSize; i++) {
    float binVal = hist.at<float>(i);
    int intensity = cvRound(binVal * height / maxVal);
    rectangle(histPlot, Point(i * binWidth, height),
              Point((i + 1) * binWidth, height - intensity),
              Scalar(255, 255, 255), -1);
  }

  // mostrar la imagen y el histograma
  imshow("Histogram", histPlot);
  return 0;
}

int get_segments(const Mat &src, const Mat &cdf, vector<Mat> &segments) {
  int segm[5] = {1, 0, 0, 0, 256};
  for (int i = 1; i < 256; i++) {
    float value_tmp = cdf.at<float>(i);
    if (value_tmp < 0.25)
      segm[1] = i;
    if (value_tmp == segm[1] || value_tmp < 0.50)
      segm[2] = i;
    if (value_tmp == segm[2] || value_tmp < 0.75)
      segm[3] = i;
  }

  for (int i = 1; i < 5; i++) {
    Mat tmp = Mat::zeros(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
      uchar *p_dst = (uchar *)tmp.ptr<uchar>(r);
      uchar *p_src = (uchar *)src.ptr<uchar>(r);
      for (int c = 0; c < src.cols; c++) {
        if (p_src[c] >= segm[i - 1] && p_src[c] < segm[i])
          p_dst[c] = 255;
      }
    }
    segments.push_back(tmp);
  }
  return 0;
}

int color_transfer(const Mat &src, const Mat &src_segment, const Mat &target,
                   const Mat &target_segment, Mat &output) {
  Mat src_tmp, target_tmp;
  get_real_segment(src, src_segment, src_tmp);
  get_real_segment(target, target_segment, target_tmp);

  src_tmp.copyTo(output);

  float src_mean = get_mean(src_tmp);
  float src_stdev = get_stdev(src_tmp, src_mean);
  float target_mean = get_mean(target_tmp);
  float target_stdev = get_stdev(target_tmp, target_mean);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_msk = (uchar *)src_segment.ptr<uchar>(r);
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    uchar *p_dst = (uchar *)output.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      if (p_msk[c]) {
        p_dst[c] =
            (target_stdev / src_stdev) * (p_src[c] - src_mean) + target_mean;
      }
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
  return mean / (src.rows * src.cols);
}

float get_stdev(const Mat &src, float mean) {
  float stdev = 0.0;
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      stdev += ((p_src[c] - mean) * (p_src[c] - mean));
    }
  }
  return sqrt(stdev / (src.rows * src.cols));
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
      color_transfer(src_channels[j], src_segments[i], target_channels[j],
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

int get_real_segment(const Mat &src, const Mat &msk, Mat &real_segment) {
  real_segment = Mat::zeros(src.size(), CV_8UC1);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_msk = (uchar *)msk.ptr<uchar>(r);
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    uchar *p_dst = (uchar *)real_segment.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      if (p_msk[c]) {
        p_dst[c] = p_src[c];
      }
    }
  }
  return 0;
}

float get_euclidean_distance(const Mat &src, const Mat &target) {
  int histSize = 256;       // number of bins in the histogram
  float range[] = {0, 256}; // range of pixel values
  const float *histRange = {range};

  // calculate the histogram
  Mat hist_src, hist_target;
  int channels[] = {0, 1, 2};
  calcHist(&src, 1, channels, Mat(), hist_src, 1, &histSize, &histRange, true,
           true);
  calcHist(&target, 1, channels, Mat(), hist_target, 1, &histSize, &histRange,
           true, true);

  // show_histo(hist_src, histSize);

  float distance = 0;
  for (int i = 1; i < histSize; i++) {
    float tmp = hist_src.at<uchar>(i) - hist_target.at<uchar>(i);
    distance += (tmp * tmp);
  }

  return sqrt(distance);
}

int claheImage(const Mat &src, Mat &dst) {
  Ptr<CLAHE> cobj = createCLAHE(1.0, Size(33, 33));
  cobj->apply(src, dst);
  return 0;
}

int get_edges(const Mat &src, Mat &edges) {
  int ksize = 1;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  Mat src_tmp;
  // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
  GaussianBlur(src, src_tmp, Size(3, 3), 0, 0, BORDER_DEFAULT);
  // Convert the image to grayscale
  cvtColor(src_tmp, src_tmp, COLOR_BGR2GRAY);
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Sobel(src_tmp, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
  Sobel(src_tmp, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
  // converting back to CV_8U
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
  cvtColor(edges, edges, COLOR_GRAY2BGR);

  // Dilation
  int dilation_size = 2;
  Mat element = getStructuringElement(
      MORPH_ELLIPSE, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
      Point(dilation_size, dilation_size));

  dilate(edges, edges, element);
  return 0;
}
