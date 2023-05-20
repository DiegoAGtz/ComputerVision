#include <cmath>
#include <core.hpp>
#include <highgui.hpp>
#include <imgcodecs.hpp>
#include <imgproc.hpp>
#include <iostream>
#include <opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int get_sdh(const Mat &src, Mat &sh, Mat &dh, int d, int theta);

const string img_source = "./images/test.jpg";
const Mat example =
    (Mat_<uchar>(5, 5) << 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0,
     255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 0, 255);

int main(void) {
  Mat input = imread(img_source, IMREAD_GRAYSCALE);
  Mat sh, dh;
  get_sdh(example, sh, dh, 2, 0);
  cout << "Sums" << endl << sh << endl;
  cout << "Sums" << endl << dh << endl;
  cout << "Sums - total elements" << endl << sum(sh)[0] << endl;
  cout << "Diffs - total elements" << endl << sum(dh)[0] << endl;
  imshow("Sums", sh);
  imshow("Differences", dh);
  waitKey(0);
  return 0;
}

int get_sdh(const Mat &src, Mat &sh, Mat &dh, int d, int theta) {
  int histSize = 511; // number of bins in the histogram

  // calculate the histogram
  sh = Mat::zeros(1, histSize, CV_16UC1);
  dh = Mat::zeros(1, histSize, CV_16UC1);
  for (int r = 0; r < src.rows; r++) {
    uchar *p_src = (uchar *)src.ptr<uchar>(r);
    ushort *p_shist = (ushort *)sh.ptr<ushort>(0);
    ushort *p_dhist = (ushort *)dh.ptr<ushort>(0);
    for (int c = 0; c < src.cols - 1; c++) {
      p_shist[p_src[c] + p_src[c + 1]]++;
      p_dhist[p_src[c] - p_src[c + 1] + 255]++;
    }
  }
  return 0;
}
