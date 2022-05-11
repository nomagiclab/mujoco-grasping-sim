#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

//receives a 2D array of bools (true - success or false - fail) and its sizes
void display_map(bool **arr, int height, int width){
   //Crete the image base
   Mat image = Mat::zeros(height, width, CV_8U);

   //Check for failure
   if (image.empty()) 
   {
      printf("Could not create the image!!\n");
      return;
   }
   
   // Put correct values of pixels
   for(int i = 0; i < height; ++i){
      for(int j = 0; j < width; ++j){
         image.at<unsigned char>(i, j) = 255 * arr[i][j];
      }
   }

   //Display the image
   String windowName = "Successful_catches";
   namedWindow(windowName);
   imshow(windowName, image);

   waitKey(0);
   destroyWindow(windowName);

   return;
}

int main(int argc, char** argv)
{
   int H = 600;
   int W = 600;

   bool **array;
   array = new bool *[H];

   for(int i = 0; i < H; ++i){
      array[i] = new bool[W];

      for(int j = 0; j < W; ++j){
         array[i][j] = {false};
      }
   }

   array[30][10] = true;
   array[15][10] = true;

   for(int i = 40; i < 60; ++i){
      for(int j = 40; j < 50; ++j){
         array[i][j] = true;
      }
   }

   display_map(array, H, W);

   return 0;
}