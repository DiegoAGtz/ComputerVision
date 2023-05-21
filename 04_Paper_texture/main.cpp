#include <cstdlib>
#include <fstream>
#include <imgcodecs.hpp>
#include <iostream>

#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>

using namespace cv;
using namespace std;

int getSDH(const Mat &src, Mat &sh, Mat &dh, int d);
int normHist(const Mat &src, Mat &dst);
float getHomogeneity(const Mat &src);
int getDisplacementVectors(const Mat &src, Mat &hdst, Mat &vdst);
int writeCSV(string filename, Mat m);
int getTexelSize(const Mat &dsvect);

const string img_source = "./images/test4.jpg";

int main(void) {
  Mat input = imread(img_source, IMREAD_GRAYSCALE);
  Mat inputColor = imread(img_source, IMREAD_COLOR);
  Mat hdv, vdv;
  getDisplacementVectors(input, hdv, vdv);

  double htexels = getTexelSize(hdv);
  double vtexels = getTexelSize(vdv);

  cout << "Horizontal texel: " << htexels << endl;
  cout << "Vertical texel: " << vtexels << endl;

  rectangle(inputColor, Point(0, 0), Point(htexels, vtexels), Scalar(0, 0, 255),
            2);
  rectangle(inputColor, Point(htexels, 0), Point(htexels * 2, vtexels),
            Scalar(0, 0, 255), 2);
  rectangle(inputColor, Point(0, vtexels), Point(htexels, vtexels * 2),
            Scalar(0, 0, 255), 2);
  rectangle(inputColor, Point(htexels, vtexels),
            Point(htexels * 2, vtexels * 2), Scalar(0, 0, 255), 2);

  if (inputColor.rows > 1000 || inputColor.cols > 1000)
    resize(inputColor, inputColor, Size(), 0.5, 0.5);
  imshow("Texel", inputColor);
  waitKey(0);
  return 0;
}

int getSDH(const Mat &src, Mat &sh, Mat &dh, int d) {
  int histSize = 511; // number of bins in the histogram
  // calculate the histogram
  sh = Mat::zeros(1, histSize, CV_16UC1);
  dh = Mat::zeros(1, histSize, CV_16UC1);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    ushort *p_shist = (ushort *)sh.ptr<ushort>(0);
    ushort *p_dhist = (ushort *)dh.ptr<ushort>(0);
    for (int c = 0; c < src.cols - d; c++) {
      p_shist[p_src[c] + p_src[c + d]]++;
      p_dhist[p_src[c] - p_src[c + d] + 255]++;
    }
  }
  return 0;
}

int normHist(const Mat &src, Mat &dst) {
  int A = sum(src)[0];
  src.copyTo(dst);
  dst.convertTo(dst, CV_32FC1);
  dst /= A;
  return 0;
}

float getHomogeneity(const Mat &src) {
  float homogeneity = 0;
  float *row = (float *)src.ptr<float>(0);
  for (int c = 0; c < src.cols; c++) {
    homogeneity += (1.0 / (1 + ((c - 255) * (c - 255))) * row[c]);
  }
  return homogeneity;
}

int getDisplacementVectors(const Mat &src, Mat &hdst, Mat &vdst) {
  hdst = Mat(1, src.cols, CV_64FC1);
  vdst = Mat(1, src.rows, CV_64FC1);

  ofstream horHomogeneityFile("out/yo_horHomogeneity.csv", ios::out),
      verHomogeneityFile("out/yo_verHomogeneity.csv", ios::out);

  Mat sh, dh;
  Mat hdata_x(1, src.cols, CV_64FC1);
  double *dst = (double *)hdst.ptr<double>(0);
  for (int d = 1; d < src.cols; d++) {
    getSDH(src, sh, dh, d);
    normHist(dh, dh);
    dst[d] = getHomogeneity(dh);
    horHomogeneityFile << dst[d] << ((d + 1) < src.cols ? "," : "");
  }

  Mat srcpy;
  src.copyTo(srcpy);
  transpose(srcpy, srcpy);
  Mat vdata_x = Mat(1, srcpy.cols, CV_64FC1);
  dst = (double *)vdst.ptr<double>(0);
  for (int d = 1; d < srcpy.cols; d++) {
    getSDH(srcpy, sh, dh, d);
    normHist(dh, dh);
    dst[d] = getHomogeneity(dh);
    verHomogeneityFile << dst[d] << ((d + 1) < srcpy.cols ? "," : "");
  }

  horHomogeneityFile.close();
  verHomogeneityFile.close();
  return 0;
}

int writeCSV(string filename, Mat m) {
  ofstream myfile;
  myfile.open(filename.c_str());
  myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
  myfile.close();
  return 0;
}

int getTexelSize(const Mat &dsvect) {
  double *p = (double *)dsvect.ptr<double>(0);
  double max = p[10];
  int ret = 10;
  bool up = false;
  for (int c = 10; c < int(dsvect.cols / 2) - 1; c++) {
    if (p[c] > max) {
      max = p[c];
      ret = c;
      up = true;
    }
    if (up && p[c] < max) {
      cout << p[c] << " - " << max << " - " << c << endl;
      break;
    }
  }
  return ret;
}
