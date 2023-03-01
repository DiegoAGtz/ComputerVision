#include <core.hpp>
#include <highgui.hpp>
#include <imgcodecs.hpp>
#include <imgproc.hpp>
#include <iostream>

#define MAX_ROWS 50
#define MAX_COLS 100

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  cout << "Programa inicializado" << endl;
  Mat input = imread("./cvImages/0010.jpg", IMREAD_GRAYSCALE);
  Mat blur;
  medianBlur(input, blur, 5);

  Mat output;
  adaptiveThreshold(input, output, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
                    11, 2.0);
  imshow("Output: Adaptive tresh mean", output);
  waitKey(0);
  return 0;
}
