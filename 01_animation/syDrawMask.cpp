#include <iostream>
using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

int syDrawMask(void);
int syImDrawMask_Rectangular(const Mat &src, Mat &dst, Point topLeft,
                             Point bottomRight);
int syImDrawMask_Circular(const Mat &src, Mat &dst, Point center, int radius);
int syImDrawMask_Conical(const Mat &src, Mat &dst, Point center, int radius);

int syImMasking(const Mat &src, const Mat &msk, Mat &dst);
int syMinimum(int A, int B);
int syMaximum(int A, int B);

int syImMaskingRectangular(const Mat &src, Mat &dst, Point topLeft,
                           Point bottomRight);
int syImMaskingCircular(const Mat &src, Mat &dst, Point center, int radius);
int syImMaskingConical(const Mat &src, Mat &dst, Point center, int radius);

int getDirection(int direction, int position, int limit);

int main(void) {
  syDrawMask();
  return 0;
}

int syDrawMask(void) {
  cout << "\n\nUsing masks on images.\n";

  Mat Input;
  Input = imread("./cvImages/mapa.jpg",
                 IMREAD_COLOR); // Directory and file must exist!!!
  if (!Input.data) {
    cout << "\nWrong filename or folder.\n" << endl;
    return -1;
  }

  imshow("Input", Input);

  // Image rotation
  // --------------------------------------------------------------------------
  Mat image;
  Input.copyTo(image);
  double angle = 45;

  // get the center coordinates of the image to create the 2D rotation matrix
  Point2f centerf((image.cols - 1) / 2.0, (image.rows - 1) / 2.0);
  // using getRotationMatrix2D() to get the rotation matrix
  Mat rotation_matix;
  rotation_matix = getRotationMatrix2D(centerf, angle, 1.0);
  // rotationMatrix(rotation_matix, centerf, angle);

  // we will save the resulting image in rotated_image matrix
  Mat rotated_image;
  // rotate the image using warpAffine
  warpAffine(image, rotated_image, rotation_matix, image.size());
  imshow("Rotated image", rotated_image);
  // --------------------------------------------------------------------------

  Mat dsttemp;
  Point topLeft(0.25 * Input.cols, 0.35 * Input.rows),
      bottomRight(0.75 * Input.cols, 0.65 * Input.rows);
  syImMaskingRectangular(Input, dsttemp, topLeft, bottomRight);
  imshow("Rectangular", dsttemp);

  // Notice in Point (x,y) coordinates correspond to (column,row) location
  Point center(Input.cols / 2, Input.rows / 2);
  int radius = Input.rows / 5;

  int x = Input.cols / 2, y = Input.rows / 2, signoX = 1, signoY = 1;
  Mat dst;

  while (true) {

    if (x >= (Input.cols - radius) || x <= (0 + radius))
      signoX = signoX * -1;
    if (y >= (Input.rows - radius) || y <= (0 + radius))
      signoY = signoY * -1;

    x += signoX;
    y += signoY;

    center.x = x;
    center.y = y;
    syImMaskingCircular(Input, dst, center, radius);

    centerf.x = x;
    centerf.y = y;

    (angle < 360) ? angle++ : angle = 0;
    rotation_matix = getRotationMatrix2D(centerf, angle, 1.0);

    // we will save the resulting image in rotated_image matrix
    rotated_image.release();
    // rotate the image using warpAffine
    warpAffine(dst, rotated_image, rotation_matix, dst.size());

    imshow("Masked in a circle", rotated_image);

    if (waitKey(10) >= 0)
      break;
  };

  return 0;
}

int getDirection(int direction, int position, int limit) {
  if (direction > 0)
    return position >= limit ? -1 : 1;
  return position <= limit ? 1 : -1;
}

int syImMaskingConical(const Mat &src, Mat &dst, Point center, int radius) {
  // if (!src.data || src.channels() != 3 || src.type() != CV_8UC3) {
  //   cout << "syImMaskingConical(): invalid src image.";
  //   return -1;
  // }
  // if (dst.data) {
  //   cout << "syImMaskingConical(): dst image is not empty.";
  //   return -1;
  // }

  Mat msk;
  syImDrawMask_Conical(src, msk, center, radius);
  syImMasking(src, msk, dst);
  return 0;
}

int syImMaskingCircular(const Mat &src, Mat &dst, Point center, int radius) {
  //   if (!src.data || src.channels() != 3 || src.type() != CV_8UC3) {
  //     cout << "syImMaskingCircular(): invalid src image.";
  //     return -1;
  //   }
  //   if (dst.data) {
  //     cout << "syImMaskingCircular(): dst image is not empty.";
  //     return -1;
  //   }

  Mat msk;
  syImDrawMask_Circular(src, msk, center, radius);
  syImMasking(src, msk, dst);
  return 0;
}

int syImMaskingRectangular(const Mat &src, Mat &dst, Point topLeft,
                           Point bottomRight) {
  if (!src.data || src.channels() != 3 || src.type() != CV_8UC3) {
    cout << "syImMaskingRectangular(): invalid src image.";
    return -1;
  }
  if (dst.data) {
    cout << "syImMaskingRectangular(): dst image is not empty.";
    return -1;
  }

  Mat msk;
  syImDrawMask_Rectangular(src, msk, topLeft, bottomRight);
  syImMasking(src, msk, dst);
  return 0;
}

int syImMasking(const Mat &src, const Mat &msk, Mat &dst) {
  //  if (!src.data || src.channels() != 3 || src.type() != CV_8UC3) {
  //    cout << "syImMasking(): invalid src image.";
  //    return -1;
  //  }
  //  if (!msk.data || msk.channels() != 1 || msk.type() != CV_8UC1) {
  //    cout << "syImMasking(): invalid msk image.";
  //    return -1;
  //  }
  //  if (dst.data) {
  //    cout << "syImMasking(): dst image is not empty.";
  //    return -1;
  //  }

  //  if (src.rows != msk.rows || src.cols != msk.cols) {
  //    cout << "syImMasking(): src and msk images must be the same size.";
  //    return -1;
  //  }

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

int syImDrawMask_Conical(const Mat &src, Mat &dst, Point center, int radius) {
  if (!src.data) {
    cout << "syImDrawMask_Rectangular(): invalid src image.";
    return -1;
  }
  if (dst.data) {
    cout << "syImDrawMask_Rectangular(): dst image is not empty.";
    return -1;
  }

  dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

  int centerR = center.y; // Notice in Point (x,y) coordinates correspond to
                          // (column,row) location
  int centerC = center.x;

  int lowR = syMaximum(centerR - radius, 0);
  int uppR = syMinimum(centerR + radius, dst.rows - 1);
  int lowC = syMaximum(centerC - radius, 0);
  int uppC = syMinimum(centerC + radius, dst.cols - 1);

  for (int r = lowR; r < uppR; r++) {
    uchar *pDst = dst.ptr<uchar>(r);
    for (int c = lowC; c < uppC; c++) {
      float dist = sqrt(((r - centerR) * (r - centerR)) +
                        ((c - centerC) * (c - centerC)));
      if (dist < radius)
        pDst[c] = (uchar)(255.0 * (radius - dist) / radius);
    }
  }
  return 0;
}

int syImDrawMask_Circular(const Mat &src, Mat &dst, Point center, int radius) {
  if (!src.data) {
    cout << "syImDrawMask_Circular(): invalid src image.";
    return -1;
  }
  if (dst.data) {
    cout << "syImDrawMask_Circular(): dst image is not empty.";
    return -1;
  }

  dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
  circle(dst, center, radius, 255, -1, LINE_8, 0);
  return 0;
}

int syImDrawMask_Rectangular(const Mat &src, Mat &dst, Point topLeft,
                             Point bottomRight) {
  if (!src.data) {
    cout << "syImDrawMask_Rectangular(): invalid src image.";
    return -1;
  }
  if (dst.data) {
    cout << "syImDrawMask_Rectangular(): dst image is not empty.";
    return -1;
  }

  dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
  rectangle(dst, topLeft, bottomRight, 255, -1,
            LINE_8); // thickness -1 is the key
  return 0;
}

int syMinimum(int A, int B) {
  if (A < B)
    return A;
  return B;
}

int syMaximum(int A, int B) {
  if (A > B)
    return A;
  return B;
}
