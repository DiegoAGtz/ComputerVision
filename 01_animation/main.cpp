#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

int init(void);
int drawMask(const Mat &src, const Mat &msk, Mat &dst, int radius);
int getCircularMask(const Mat &src, Mat &dst, Point center, int radius);
int rotate(const Mat &src, Mat &dst, Point2f center, double angle,
           double scale);

int readFromCSV(const string filename, Mat &dst);
int drawBoard(const Mat &src, Mat &dst);
int drawRectangle(Mat &src, int topLeftR, int topLeftC, int bottomRightR,
                  int bottomRightC, int color);

int checkNeighborhood(int row, int col, const Mat &src);
int nextGeneration(Mat &src);

int main(int argc, char *argv[]) {
  init();
  return 0;
}

int init(void) {
  Mat board, game;
  readFromCSV("tableroInicial.csv", game);
  for (int i = 0; i < 11; i++) {
    drawBoard(game, board);
    imshow("Tablero", board);
    nextGeneration(game);
    board.release();
    waitKey(1000);
  }

  // ------------------------------------------------
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

// Conway's game functions
int readFromCSV(const string filename, Mat &dst) {
  unsigned char data[50][100];
  string line, token;
  fstream file(filename, ios::in);
  if (!file.is_open()) {
    cout << "No se pudo abrir el fichero" << endl;
    return -1;
  }
  for (int i = 0; i < 50; i++) {
    getline(file, line);
    stringstream ss(line);
    for (int j = 0; j < 100; j++) {
      getline(ss, token, ',');
      data[i][j] = token.at(0) - 48;
    }
  }
  file.close();

  dst = Mat::zeros(50, 100, CV_8UC1);
  for (int r = 0; r < dst.rows; r++) {
    uchar *pAtDst = dst.ptr<uchar>(r);
    for (int c = 0; c < dst.cols; c++)
      pAtDst[c] = data[r][c];
  }
  return 0;
}

int drawBoard(const Mat &src, Mat &dst) {
  if (!dst.empty()) {
    cout << "\nsyImChessBoard(): Input must be an empty image." << endl;
    return -1;
  }

  int deltaC = 10;
  int deltaR = 10;

  dst = Mat::zeros(src.rows * deltaR, src.cols * deltaC, src.type());
  drawRectangle(dst, 0, 0, dst.rows, dst.cols, 0);

  for (int r = 0; r < src.rows; r++) {
    uchar *pAtSrc = (uchar *)src.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      if (pAtSrc[c]) {
        drawRectangle(dst, r * deltaR, c * deltaC, (r + 1) * deltaR,
                      (c + 1) * deltaC, 255);
      }
    }
  }
  return 0;
}

int drawRectangle(Mat &src, int topLeftR, int topLeftC, int bottomRightR,
                  int bottomRightC, int color) {

  if (src.empty()) {
    cout << "There is no image" << endl;
    return -1;
  }

  if (topLeftC < 0)
    topLeftC = 0;
  if (bottomRightC > src.cols)
    bottomRightC = src.cols;
  if (topLeftR < 0)
    topLeftR = 0;
  if (bottomRightR > src.rows)
    bottomRightR = src.rows;
  if (topLeftC > bottomRightC || topLeftR > bottomRightR) {
    cout << " Incorrect set of coordinates" << endl;
    return -1;
  }

  rectangle(src, Point(topLeftC, topLeftR), Point(bottomRightC, bottomRightR),
            color, -1, LINE_8);

  return 0;
}

int checkNeighborhood(int row, int col, const Mat &src) {
  int counter = 0;
  uchar *pAtSrcBefore, *pAtSrcActual, *pAtSrcAfter;

  if (row > 0)
    pAtSrcBefore = (uchar *)src.ptr<uchar>(row - 1);
  if (row < src.rows)
    pAtSrcAfter = (uchar *)src.ptr<uchar>(row + 1);
  pAtSrcActual = (uchar *)src.ptr<uchar>(row);

  // Check corners
  if (row > 0 && col > 0 && pAtSrcBefore[col - 1])
    counter++;
  if (row > 0 && col < src.cols - 1 && pAtSrcBefore[col + 1])
    counter++;
  if (row < src.rows - 1 && col > 0 && pAtSrcAfter[col - 1])
    counter++;
  if (row < src.rows - 1 && col < src.cols - 1 && pAtSrcAfter[col + 1])
    counter++;

  // Check edge
  if (row > 0 && pAtSrcBefore[col])
    counter++;
  if (row < src.rows - 1 && pAtSrcAfter[col])
    counter++;
  if (col > 0 && pAtSrcActual[col - 1])
    counter++;
  if (col < src.cols - 1 && pAtSrcActual[col + 1])
    counter++;

  // pAtSrcActual[col] = 0;
  // cout << "Numer of ocurrences at (" << row << "," << col << "): " << counter
  //      << endl;
  return counter;
}

int nextGeneration(Mat &src) {
  int counter = 0;
  Mat dst = src.clone();
  for (int r = 0; r < src.rows; r++) {
    uchar *pAtSrc = src.ptr<uchar>(r);
    uchar *pAtDst = dst.ptr<uchar>(r);
    for (int c = 0; c < src.cols; c++) {
      counter = checkNeighborhood(r, c, src);
      // Rules
      if (counter == 3 || (counter == 2 && pAtSrc[c]))
        pAtDst[c] = 1;
      if (counter > 3)
        pAtDst[c] = 0;
      if (counter <= 1)
        pAtDst[c] = 0;
    }
  }
  src = dst.clone();
  return 0;
}
