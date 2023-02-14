#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int init(void);
int drawMask(const Mat &src, const Mat &msk, Mat &dst, int radius);
int getCircularMask(const Mat &src, Mat &dst, Point center, int radius);
int rotate(const Mat &src, Mat &dst, Point2f center, double angle,
           double scale);

int main(int argc, char *argv[]) {
  init();
  return 0;
}

int init(void) {
  Mat input = imread("./cvImages/mapa.jpg", IMREAD_COLOR);
  if (!input.data) {
    cout << "Ruta de imagen incorrecta" << endl;
    return -1;
  }
  imshow("Input", input);

  Point center(input.cols / 2, input.rows / 2);
  Mat mask, dst, rotated;
  double angle = 0, scale = 1.0;
  int radius = input.rows / 4, directionX = 1, directionY = 1;

  while (true) {
    mask.release();
    dst.release();
    rotated.release();

    if (center.x >= input.cols - radius || center.x <= 0 + radius)
      directionX *= -1;
    if (center.y >= input.rows - radius || center.y <= 0 + radius)
      directionY *= -1;

    center.x += directionX;
    center.y += directionY;

    getCircularMask(input, mask, center, radius);
    drawMask(input, mask, dst, radius);

    (angle < 360) ? angle++ : angle = 0;
    rotate(dst, rotated, (Point2f)center, angle, scale);
    imshow("Masked in a Circle", rotated);

    if (waitKey(10) >= 0)
      break;
  }

  return 0;
}

int drawMask(const Mat &src, const Mat &msk, Mat &dst, int radius) {
  if (!src.data || src.channels() != 3 || src.type() != CV_8UC3) {
    cout << "drawMask(): invalid src image.";
    return -1;
  }
  if (!msk.data || msk.channels() != 1 || msk.type() != CV_8UC1) {
    cout << "drawMask(): invalid msk image.";
    return -1;
  }
  if (dst.data) {
    cout << "drawMask(): dst image is not empty.";
    return -1;
  }
  if (src.rows != msk.rows || src.cols != msk.cols) {
    cout << "drawMask(): src and msk images must be the same size.";
    return -1;
  }

  dst = src.clone();

  for (int r = 0; r < dst.rows; r++) {
    Vec3b *pDst = dst.ptr<Vec3b>(r);
    uchar *pMsk = (uchar *)msk.ptr<uchar>(r);

    for (int c = 0; c < dst.cols; c++) {
      pDst[c][0] *= (pMsk[c] / 255.0);
      pDst[c][1] *= (pMsk[c] / 255.0);
      pDst[c][2] *= (pMsk[c] / 255.0);
    }
  }
  return 0;
}

int getCircularMask(const Mat &src, Mat &dst, Point center, int radius) {
  if (!src.data) {
    cout << "getCircularMask(): La imagen de entrada no es valida." << endl;
    return -1;
  }
  if (dst.data) {
    cout << "getCircularMask(): La imagen de destino no esta vacia." << endl;
    return -1;
  }
  // dst = Mat::zeros(src.rows, src.cols, src.type());
  dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
  circle(dst, center, radius, 255, -1, LINE_8, 0);
  return 0;
}

int rotate(const Mat &src, Mat &dst, Point2f center, double angle,
           double scale) {
  Mat rotation_matrix = getRotationMatrix2D(center, angle, scale);
  warpAffine(src, dst, rotation_matrix, src.size());
  return 0;
}
